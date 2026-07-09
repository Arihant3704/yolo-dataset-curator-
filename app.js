/**
 * CleanCrop - Dataset Quality & Curation Dashboard JavaScript Engine
 * Core Functionality:
 * 1. Embeddings Projection via Web PCA Power Method
 * 2. Real-time Curation Metrics Re-computation
 * 3. Interactive Embedding Space Canvas Plot (Pan, Zoom, Hover, Select)
 * 4. Human-in-the-Loop Curation Actions (Exclude, Relabel, Keep)
 * 5. Dynamic Report JSON Upload and Curation Export
 */

(function () {
    // APP STATE
    let reportData = null;      // Holds the active dataset curation report
    let activeTab = 'overview'; // Active dashboard navigation tab
    
    // User Curation State (Manual Overrides)
    const deletedIds = new Set();               // Excluded crops
    const relabeledLabels = new Map();          // Map of ID -> New Class Label String
    const keptIds = new Set();                  // Crops manually approved (clears outlier/mislabel flags)
    const ignoredDuplicates = new Set();        // Set of "idA-idB" duplicate pairs marked false positive
    const actionsLog = [];                      // Array of changes made for auditing
    
    // Algorithmic Parameters (Sliders)
    let dupThreshold = 0.95;
    let outCentroidThreshold = 0.85;
    let outKnnThreshold = 0.50;
    let mislabelK = 5;
    let mislabelConsensus = 0.50; // Ratio of same-label neighbors
    
    // PCA / Plot State
    let pcaCoordinates = []; // Projected 2D coordinates [[x, y], ...]
    let classColors = {};
    let defaultColor = '#58a6ff';
    
    // Canvas View State
    const canvas = document.getElementById('embedding-canvas');
    const ctx = canvas.getContext('2d');
    let viewZoom = 1.0;
    let viewPanX = 0.0;
    let viewPanY = 0.0;
    let isDrawing = false;
    let dragStart = { x: 0, y: 0 };
    let dragPanStart = { x: 0, y: 0 };
    let hoveredIndex = -1;
    let selectedIndex = -1;
    let showAnomalyRings = true;
    
    // 3D Mode State
    let is3D = false;
    let rotationX = -0.5;
    let rotationY = 0.6;
    let dragRotationStart = { x: 0, y: 0 };
    
    // Initial Load
    window.addEventListener('DOMContentLoaded', () => {
        // 1. Check if global CURATION_REPORT is loaded from file
        if (window.CURATION_REPORT) {
            loadDataset(window.CURATION_REPORT);
        } else {
            showErrorState("No dataset loaded. Please upload a curation JSON report.");
        }
        
        // 2. Set up Nav Event Listeners
        document.querySelectorAll('.menu-item').forEach(item => {
            item.addEventListener('click', () => {
                switchTab(item.getAttribute('data-tab'));
            });
        });
        
        // 3. Set up Slider Listeners
        setupSliders();
        
        // 4. Set up Canvas Event Listeners
        setupCanvas();
        
        // 5. File Upload Handler
        document.getElementById('json-upload').addEventListener('change', handleFileUpload);
        
        // 6. Action Button Handlers
        setupActionButtons();
    });

    // ==========================================================================
    // DATA LOADING & COMPUTE PIPELINE
    // ==========================================================================
    
    function loadDataset(report) {
        reportData = report;
        
        // Set initial slider parameters matching report
        if (report.summary) {
            dupThreshold = report.summary.duplicate_threshold || 0.95;
            outCentroidThreshold = report.summary.outlier_threshold_centroid || 0.85;
            outKnnThreshold = report.summary.outlier_threshold_knn;
            if (!outKnnThreshold || outKnnThreshold > 1.0) {
                outKnnThreshold = 0.50;
            }
            mislabelK = report.summary.mislabel_k || 5;
            
            document.getElementById('slider-dup').value = dupThreshold;
            document.getElementById('val-dup').textContent = dupThreshold.toFixed(2);
            document.getElementById('slider-out-centroid').value = outCentroidThreshold;
            document.getElementById('val-out-centroid').textContent = outCentroidThreshold.toFixed(2);
            document.getElementById('slider-mis-k').value = mislabelK;
            document.getElementById('val-mis-k').textContent = mislabelK;
        }
        
        // Extract Unique Classes from items
        const classes = [...new Set(reportData.items.map(item => item.label))];
        
        // Clear classColors and dynamically assign class colors
        classColors = {};
        const hues = [0, 45, 135, 200, 275, 310, 25, 75, 170, 225, 290, 340];
        classes.forEach((cls, idx) => {
            classColors[cls] = `hsl(${hues[idx % hues.length]}, 85%, 60%)`;
        });
        
        // Populate Filter dropdowns
        const filterClass = document.getElementById('filter-class');
        filterClass.innerHTML = '<option value="all">All Classes</option>';
        classes.forEach(cls => {
            filterClass.innerHTML += `<option value="${cls}">${cls}</option>`;
        });
        
        const expRelabelSelect = document.getElementById('exp-relabel-select');
        expRelabelSelect.innerHTML = '';
        classes.forEach(cls => {
            expRelabelSelect.innerHTML += `<option value="${cls}">${cls}</option>`;
        });
        
        // Run PCA on high-dim embeddings
        console.log("Computing PCA Projection in browser...");
        const t0 = performance.now();
        if (reportData.embeddings && reportData.embeddings.length > 0) {
            pcaCoordinates = computePCA(reportData.embeddings, 3);
        } else {
            // Fallback synthetic PCA if embeddings are missing
            pcaCoordinates = reportData.items.map((item, idx) => {
                const angle = (idx / reportData.items.length) * Math.PI * 2;
                return [Math.cos(angle) * 2, Math.sin(angle) * 2, Math.sin(angle * 2) * 0.5];
            });
        }
        console.log(`PCA computed in ${(performance.now() - t0).toFixed(1)}ms.`);
        
        // Reset manual curation logs
        deletedIds.clear();
        relabeledLabels.clear();
        keptIds.clear();
        ignoredDuplicates.clear();
        actionsLog.length = 0;
        selectedIndex = -1;
        hoveredIndex = -1;
        
        // Run Curation Engine
        recomputeCuration();
        
        // Fit canvas viewpoint to data
        resetCanvasView();
    }
    
    /**
     * SVD/PCA Projection using Power Iteration Method
     * Projects N x D normalized embeddings to N x 2 space
     */
    function computePCA(X, numComponents = 2) {
        const N = X.length;
        const D = X[0].length;
        
        // 1. Center the data
        const mean = new Array(D).fill(0);
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < D; j++) {
                mean[j] += X[i][j];
            }
        }
        for (let j = 0; j < D; j++) mean[j] /= N;
        
        const Xc = [];
        for (let i = 0; i < N; i++) {
            Xc.push(X[i].map((val, j) => val - mean[j]));
        }
        
        // Matrix-vector multiplication for covariance estimate: Xc^T * (Xc * v)
        function covarianceProduct(v) {
            const Xv = new Array(N).fill(0);
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < D; j++) {
                    Xv[i] += Xc[i][j] * v[j];
                }
            }
            const XT_Xv = new Array(D).fill(0);
            for (let j = 0; j < D; j++) {
                for (let i = 0; i < N; i++) {
                    XT_Xv[j] += Xc[i][j] * Xv[i];
                }
            }
            return XT_Xv;
        }
        
        function normalize(v) {
            const len = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
            return len > 1e-9 ? v.map(val => val / len) : v;
        }
        
        const pcs = [];
        let tempX = Xc;
        
        for (let c = 0; c < numComponents; c++) {
            let v = new Array(D).fill(0).map(() => Math.random() - 0.5);
            v = normalize(v);
            
            // Power Method iterations
            for (let iter = 0; iter < 12; iter++) {
                v = covarianceProduct(v);
                v = normalize(v);
            }
            pcs.push(v);
            
            // Deflate Matrix to find the next orthogonal component
            if (c < numComponents - 1) {
                for (let i = 0; i < N; i++) {
                    let proj = 0;
                    for (let j = 0; j < D; j++) proj += Xc[i][j] * v[j];
                    for (let j = 0; j < D; j++) Xc[i][j] -= proj * v[j];
                }
            }
        }
        
        // Project centered points onto PCs
        const projected = [];
        for (let i = 0; i < N; i++) {
            const row = [];
            for (let c = 0; c < numComponents; c++) {
                let sum = 0;
                for (let j = 0; j < D; j++) {
                    sum += Xc[i][j] * pcs[c][j];
                }
                row.push(sum);
            }
            projected.push(row);
        }
        
        return projected;
    }
    
    /**
     * CORE ALGORITHMIC ENGINE
     * Performs vector operations to flag duplicates, outliers, and mislabeled crops.
     * Computes dynamically in-browser when slider configurations change.
     */
    function recomputeCuration() {
        if (!reportData) return;
        
        const t0 = performance.now();
        const items = reportData.items;
        const embs = reportData.embeddings;
        const N = items.length;
        const D = (embs && embs.length > 0) ? embs[0].length : 128;
        
        // 1. Reset all active flags
        items.forEach(item => {
            item.is_duplicate = false;
            item.is_outlier = false;
            item.is_mislabeled = false;
            item.duplicate_pairs = [];
            
            // Get curbed status
            item.curLabel = relabeledLabels.get(item.id) || item.label;
            item.curDeleted = deletedIds.has(item.id);
            item.curKept = keptIds.has(item.id);
        });
        
        // 2. DUPLICATE DETECTION (All pairs cosine similarity)
        const activeDuplicatePairs = [];
        const dupSet = new Set();
        
        for (let i = 0; i < N; i++) {
            if (items[i].curDeleted) continue;
            for (let j = i + 1; j < N; j++) {
                if (items[j].curDeleted) continue;
                
                // Cosine Similarity = dot product of normalized embeddings
                let sim = 0;
                for (let d = 0; d < D; d++) {
                    sim += embs[i][d] * embs[j][d];
                }
                
                if (sim >= dupThreshold) {
                    const pairKey = items[i].id < items[j].id ? `${items[i].id}-${items[j].id}` : `${items[j].id}-${items[i].id}`;
                    if (ignoredDuplicates.has(pairKey)) continue;
                    
                    items[i].is_duplicate = true;
                    items[j].is_duplicate = true;
                    
                    items[i].duplicate_pairs.push({ index: j, sim: sim });
                    items[j].duplicate_pairs.push({ index: i, sim: sim });
                    
                    dupSet.add(i);
                    dupSet.add(j);
                    
                    activeDuplicatePairs.push({
                        pairKey: pairKey,
                        a: items[i],
                        b: items[j],
                        indexA: i,
                        indexB: j,
                        similarity: sim
                    });
                }
            }
        }
        
        // 3. OUTLIERS & MISLABELS (Centroids + KNN)
        // Group indices of active items by their current label
        const classGroups = {};
        for (let i = 0; i < N; i++) {
            if (items[i].curDeleted) continue;
            const lbl = items[i].curLabel;
            if (!classGroups[lbl]) classGroups[lbl] = [];
            classGroups[lbl].push(i);
        }
        
        // Calculate Class Centroids (mean vector normalized)
        const centroids = {};
        for (const [clsName, indices] of Object.entries(classGroups)) {
            if (indices.length === 0) continue;
            const centroid = new Array(D).fill(0);
            indices.forEach(idx => {
                for (let d = 0; d < D; d++) {
                    centroid[d] += embs[idx][d];
                }
            });
            // Normalize centroid
            let len = Math.sqrt(centroid.reduce((sum, val) => sum + val * val, 0));
            centroids[clsName] = len > 1e-9 ? centroid.map(val => val / len) : centroid;
        }
        
        const activeOutliers = [];
        const activeMislabels = [];
        
        // Neighbor Analysis & Outlier Scoring
        for (let i = 0; i < N; i++) {
            if (items[i].curDeleted) continue;
            
            // A. Centroid Similarity
            const myCentroid = centroids[items[i].curLabel];
            let centroidSim = 0;
            if (myCentroid) {
                for (let d = 0; d < D; d++) {
                    centroidSim += embs[i][d] * myCentroid[d];
                }
            }
            items[i].centroid_similarity = centroidSim;
            
            // B. Neighbors Query
            const sims = [];
            for (let j = 0; j < N; j++) {
                if (i === j || items[j].curDeleted) continue;
                let dot = 0;
                for (let d = 0; d < D; d++) {
                    dot += embs[i][d] * embs[j][d];
                }
                sims.push({ index: j, sim: dot, label: items[j].curLabel, id: items[j].id });
            }
            
            // Sort to get KNN nearest neighbors
            sims.sort((a, b) => b.sim - a.sim);
            const knn = sims.slice(0, mislabelK);
            
            // Calculate KNN average similarity
            let knnSum = 0;
            knn.forEach(n => knnSum += n.sim);
            const knnSim = knn.length > 0 ? knnSum / knn.length : 0;
            items[i].knn_similarity = knnSim;
            
            // C. Outlier flagging
            const isCentroidOutlier = centroidSim < outCentroidThreshold;
            const isKnnOutlier = knnSim < outKnnThreshold;
            const isOutlier = isCentroidOutlier || isKnnOutlier;
            
            if (isOutlier && !items[i].curKept) {
                items[i].is_outlier = true;
                
                let reason = "";
                if (isCentroidOutlier && isKnnOutlier) {
                    reason = `Low similarity to both centroid (${centroidSim.toFixed(2)}) and neighbors (${knnSim.toFixed(2)}).`;
                } else if (isCentroidOutlier) {
                    reason = `Low similarity to centroid (${centroidSim.toFixed(2)}). Centroid threshold: ${outCentroidThreshold.toFixed(2)}.`;
                } else {
                    reason = `Low similarity to neighbors (${knnSim.toFixed(2)}). KNN threshold: ${outKnnThreshold.toFixed(2)}.`;
                }
                
                activeOutliers.push({
                    index: i,
                    id: items[i].id,
                    label: items[i].curLabel,
                    path: items[i].path,
                    centroid_similarity: centroidSim,
                    knn_similarity: knnSim,
                    reason: reason,
                    is_centroid: isCentroidOutlier,
                    is_knn: isKnnOutlier
                });
            }
            
            // D. Mislabel checking
            let ownClassCount = 0;
            const labelCounts = {};
            
            knn.forEach(n => {
                if (n.label === items[i].curLabel) ownClassCount++;
                labelCounts[n.label] = (labelCounts[n.label] || 0) + 1;
            });
            
            const ownClassRatio = knn.length > 0 ? ownClassCount / knn.length : 1.0;
            
            // Suspected mislabeled if neighborhood label ratio falls below consensus threshold
            const isMislabeled = ownClassRatio < mislabelConsensus;
            
            if (isMislabeled && !items[i].curKept) {
                items[i].is_mislabeled = true;
                
                // Find suggested majority label
                let maxCount = -1;
                let suggested = items[i].curLabel;
                for (const [cls, count] of Object.entries(labelCounts)) {
                    if (count > maxCount) {
                        maxCount = count;
                        suggested = cls;
                    }
                }
                
                items[i].suggested_label = suggested;
                items[i].mislabel_reason = `Only ${ownClassCount}/${knn.length} nearest neighbors belong to '${items[i].curLabel}'. Suggested: '${suggested}' (${maxCount}/${knn.length} neighbors).`;
                
                activeMislabels.push({
                    index: i,
                    id: items[i].id,
                    label: items[i].curLabel,
                    path: items[i].path,
                    suggested_label: suggested,
                    reason: items[i].mislabel_reason,
                    neighbors: knn.map(n => ({
                        id: n.id,
                        label: n.label,
                        sim: n.sim,
                        path: items[n.index].path
                    }))
                });
            }
        }
        
        // 4. UPDATE STATS & BIND LISTS TO REPORT OBJECT
        reportData.duplicates = activeDuplicatePairs;
        reportData.outliers = activeOutliers;
        reportData.mislabels = activeMislabels;
        
        const duplicateCount = dupSet.size;
        const outlierCount = activeOutliers.length;
        const mislabelCount = activeMislabels.length;
        
        // Calculate clean count
        let unionIndices = new Set();
        items.forEach((item, idx) => {
            if (item.curDeleted) return;
            if (item.is_duplicate || item.is_outlier || item.is_mislabeled) {
                unionIndices.add(idx);
            }
        });
        
        const activeTotal = N - deletedIds.size;
        const cleanCount = activeTotal - unionIndices.size;
        const healthScore = activeTotal > 0 ? (cleanCount / activeTotal) * 100 : 100;
        
        // Save dynamically calculated summary
        reportData.summary = {
            total_samples: activeTotal,
            duplicate_count: duplicateCount,
            duplicate_pairs_count: activeDuplicatePairs.length,
            outlier_count: outlierCount,
            mislabel_count: mislabelCount,
            clean_count: cleanCount,
            health_score: healthScore
        };
        
        console.log(`Dynamic curation recompute completed in ${(performance.now() - t0).toFixed(1)}ms.`);
        
        // 5. UPDATE USER INTERFACE
        updateUI();
    }
    
    // ==========================================================================
    // UI UPDATES & TAB ROUTING
    // ==========================================================================
    
    function updateUI() {
        if (!reportData) return;
        
        const summary = reportData.summary;
        
        // Update Sidebar Count Badges
        document.getElementById('badge-dup-count').textContent = summary.duplicate_pairs_count;
        document.getElementById('badge-out-count').textContent = summary.outlier_count;
        document.getElementById('badge-mis-count').textContent = summary.mislabel_count;
        
        // Update Overview Cards
        document.getElementById('stat-health-score').textContent = `${summary.health_score.toFixed(1)}%`;
        document.getElementById('stat-duplicate-count').textContent = summary.duplicate_pairs_count;
        document.getElementById('stat-outlier-count').textContent = summary.outlier_count;
        document.getElementById('stat-mislabel-count').textContent = summary.mislabel_count;
        
        // Render Active Tab Content
        renderActiveTab();
        
        // Render Canvas Scatter Plot
        drawCanvas();
        
        // Update Curation Action Floating Bar
        updateExportBar();
    }
    
    function switchTab(tabId) {
        activeTab = tabId;
        
        // Update Sidebar Active state
        document.querySelectorAll('.menu-item').forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('data-tab') === tabId) {
                item.classList.add('active');
            }
        });
        
        // Update Main View Panels Visibility
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        document.getElementById(`tab-${tabId}`).classList.add('active');
        
        // Update Header Titles
        const titles = {
            'overview': { title: "Overview Dashboard", desc: "High-level insights and statistics on dataset health." },
            'duplicates': { title: "Near-Duplicate Detection", desc: "Identify leaked or redundant crop pairs leaking metadata." },
            'outliers': { title: "Outlier Crop Detector", desc: "Crops exhibiting anomalous embedding behaviors (junk/bad crops)." },
            'mislabels': { title: "Label Consensus Inspector", desc: "Crops mismatched against their embedding nearest neighbors." },
            'explorer': { title: "Interactive Crop Explorer", desc: "Search, filter, and inspect embedding representations." }
        };
        
        document.getElementById('current-tab-title').textContent = titles[tabId].title;
        document.getElementById('current-tab-desc').textContent = titles[tabId].desc;
        
        renderActiveTab();
    }
    
    function renderActiveTab() {
        if (!reportData) return;
        
        switch (activeTab) {
            case 'overview':
                renderOverviewTab();
                break;
            case 'duplicates':
                renderDuplicatesTab();
                break;
            case 'outliers':
                renderOutliersTab();
                break;
            case 'mislabels':
                renderMislabelsTab();
                break;
            case 'explorer':
                renderExplorerTab();
                break;
        }
    }
    
    // --- 1. OVERVIEW TAB ---
    function renderOverviewTab() {
        const items = reportData.items;
        
        // A. Class Distribution count
        const classCounts = {};
        let activeCount = 0;
        
        items.forEach(item => {
            if (item.curDeleted) return;
            classCounts[item.curLabel] = (classCounts[item.curLabel] || 0) + 1;
            activeCount++;
        });
        
        const listContainer = document.getElementById('class-distribution-list');
        listContainer.innerHTML = '';
        
        const sortedClasses = Object.entries(classCounts).sort((a, b) => b[1] - a[1]);
        
        sortedClasses.forEach(([clsName, count]) => {
            const pct = activeCount > 0 ? (count / activeCount) * 100 : 0;
            const color = classColors[clsName] || defaultColor;
            
            listContainer.innerHTML += `
                <div class="bar-row">
                    <div class="bar-label">
                        <span class="bar-class-name">${clsName}</span>
                        <span class="bar-count-val">${count} crops (${pct.toFixed(1)}%)</span>
                    </div>
                    <div class="bar-outer">
                        <div class="bar-inner" style="width: ${pct}%; background-color: ${color};"></div>
                    </div>
                </div>
            `;
        });
        
        // Populate Legend in Plot
        const legendContainer = document.getElementById('plot-legend');
        legendContainer.innerHTML = '';
        Object.entries(classColors).forEach(([clsName, color]) => {
            legendContainer.innerHTML += `
                <div class="legend-item">
                    <div class="legend-dot" style="background-color: ${color};"></div>
                    <span>${clsName}</span>
                </div>
            `;
        });
        
        // B. Metric Inflation Risk Meter
        const dupPct = activeCount > 0 ? (reportData.summary.duplicate_count / activeCount) * 100 : 0;
        const meterFill = document.getElementById('meter-fill');
        const meterScore = document.getElementById('meter-score');
        
        // Rotate meter representation based on duplicate ratio
        const degree = Math.min(180, (dupPct / 15) * 180) - 45;
        meterFill.style.transform = `rotate(${degree}deg)`;
        
        if (dupPct > 8) {
            meterScore.textContent = "High";
            meterScore.style.color = "var(--color-outlier)";
            meterFill.style.borderColor = "var(--color-outlier)";
        } else if (dupPct > 3) {
            meterScore.textContent = "Med";
            meterScore.style.color = "var(--color-mislabel)";
            meterFill.style.borderColor = "var(--color-mislabel)";
        } else {
            meterScore.textContent = "Low";
            meterScore.style.color = "var(--color-clean)";
            meterFill.style.borderColor = "var(--color-clean)";
        }
        
        // C. Curation Progress Tracker
        const totalAnomalies = reportData.summary.duplicate_pairs_count + reportData.summary.outlier_count + reportData.summary.mislabel_count;
        const totalResolved = actionsLog.length;
        const progressPct = totalAnomalies + totalResolved > 0 ? (totalResolved / (totalAnomalies + totalResolved)) * 100 : 100;
        
        document.getElementById('curation-progress').style.width = `${progressPct}%`;
        document.getElementById('curation-progress-text').textContent = `${totalResolved} action(s) registered`;
    }
    
    // --- 2. DUPLICATES TAB ---
    function renderDuplicatesTab() {
        const listContainer = document.getElementById('dup-list-container');
        const countTitle = document.getElementById('dup-pairs-title');
        
        const dups = reportData.duplicates || [];
        countTitle.textContent = `Duplicate Pairs (${dups.length})`;
        
        if (dups.length === 0) {
            listContainer.innerHTML = `
                <div class="empty-state-msg" style="height: 100px;">
                    <i class="fa-solid fa-copy" style="font-size:24px;"></i>
                    <p>No duplicates found at current similarity threshold.</p>
                </div>
            `;
            document.getElementById('dup-empty-state').classList.remove('hidden');
            document.getElementById('dup-inspector-content').classList.add('hidden');
            return;
        }
        
        listContainer.innerHTML = '';
        dups.forEach((pair, idx) => {
            const isASelected = selectedIndex === pair.indexA;
            const isBSelected = selectedIndex === pair.indexB;
            const isPairSelected = isASelected || isBSelected;
            
            listContainer.innerHTML += `
                <div class="item-list-card ${isPairSelected ? 'active' : ''}" data-pair-idx="${idx}">
                    <div class="list-item-meta">
                        <div class="item-thumbnail">
                            <img src="${pair.a.path}" alt="Crop A">
                        </div>
                        <div class="item-thumbnail">
                            <img src="${pair.b.path}" alt="Crop B">
                        </div>
                        <div class="item-text-info">
                            <span class="item-text-title">${pair.a.id} & ${pair.b.id}</span>
                            <span class="item-text-subtitle">Class: ${pair.a.curLabel}</span>
                        </div>
                    </div>
                    <div class="item-list-stat dup">${(pair.similarity * 100).toFixed(1)}%</div>
                </div>
            `;
        });
        
        // Add click listener on cards
        listContainer.querySelectorAll('.item-list-card').forEach(card => {
            card.addEventListener('click', () => {
                const pairIdx = parseInt(card.getAttribute('data-pair-idx'));
                selectDuplicatePair(pairIdx);
            });
        });
        
        // Keep active selection in view or show empty inspector
        const activeCard = listContainer.querySelector('.item-list-card.active');
        if (!activeCard) {
            document.getElementById('dup-empty-state').classList.remove('hidden');
            document.getElementById('dup-inspector-content').classList.add('hidden');
        }
    }
    
    function selectDuplicatePair(idx) {
        const pair = reportData.duplicates[idx];
        if (!pair) return;
        
        // Highlight active pair cards
        document.querySelectorAll('#dup-list-container .item-list-card').forEach((card, cidx) => {
            if (cidx === idx) card.classList.add('active');
            else card.classList.remove('active');
        });
        
        document.getElementById('dup-empty-state').classList.add('hidden');
        const inspector = document.getElementById('dup-inspector-content');
        inspector.classList.remove('hidden');
        
        // Bind values
        document.getElementById('dup-similarity-value').textContent = `${(pair.similarity * 100).toFixed(1)}%`;
        
        document.getElementById('dup-id-a').textContent = pair.a.id;
        document.getElementById('dup-label-a').textContent = pair.a.curLabel;
        document.getElementById('dup-img-a').src = pair.a.path;
        
        document.getElementById('dup-id-b').textContent = pair.b.id;
        document.getElementById('dup-label-b').textContent = pair.b.curLabel;
        document.getElementById('dup-img-b').src = pair.b.path;
        
        // Assign action triggers
        const btnDeleteA = inspector.querySelector('.btn-delete-crop[data-target="a"]');
        const btnDeleteB = inspector.querySelector('.btn-delete-crop[data-target="b"]');
        const btnKeepBoth = document.getElementById('btn-dup-keep-both');
        
        btnDeleteA.onclick = () => excludeCrop(pair.a.id, `Resolved duplicate pair ${pair.pairKey} by retaining ${pair.b.id}`);
        btnDeleteB.onclick = () => excludeCrop(pair.b.id, `Resolved duplicate pair ${pair.pairKey} by retaining ${pair.a.id}`);
        btnKeepBoth.onclick = () => {
            ignoredDuplicates.add(pair.pairKey);
            registerAction({
                type: 'ignore_duplicate',
                pairKey: pair.pairKey,
                a: pair.a.id,
                b: pair.b.id,
                description: `Flagged duplicate pair ${pair.pairKey} as false positive (kept both).`
            });
            recomputeCuration();
        };
    }
    
    // --- 3. OUTLIERS TAB ---
    function renderOutliersTab() {
        const gridContainer = document.getElementById('outliers-grid-container');
        const statsInfo = document.getElementById('outliers-stats');
        
        const outliers = reportData.outliers || [];
        statsInfo.textContent = `Showing ${outliers.length} flagged outlier crops`;
        
        if (outliers.length === 0) {
            gridContainer.innerHTML = `
                <div class="empty-state-msg" style="grid-column: 1/-1; height: 200px;">
                    <i class="fa-solid fa-triangle-exclamation" style="font-size: 32px;"></i>
                    <h3>No Outliers Found</h3>
                    <p>Try reducing the Centroid Similarity or KNN threshold sliders to capture less distinct outliers.</p>
                </div>
            `;
            return;
        }
        
        gridContainer.innerHTML = '';
        outliers.forEach(out => {
            const itemClassColor = classColors[out.label] || defaultColor;
            gridContainer.innerHTML += `
                <div class="outlier-card" id="out-card-${out.id}">
                    <div class="outlier-badge">OUTLIER</div>
                    <div class="crop-image-wrapper">
                        <img src="${out.path}" alt="${out.id}">
                    </div>
                    <span class="crop-id">${out.id}</span>
                    <span class="class-pill" style="background-color: ${itemClassColor}22; color: ${itemClassColor}; border: 1px solid ${itemClassColor}33">${out.label}</span>
                    
                    <p class="outlier-reason">${out.reason}</p>
                    
                    <div class="outlier-scores">
                        <div class="score-row">
                            <span class="score-lbl">Centroid Sim</span>
                            <span class="score-val ${out.is_centroid ? 'bad' : ''}">${out.centroid_similarity.toFixed(3)}</span>
                        </div>
                        <div class="score-row">
                            <span class="score-lbl">KNN Sim</span>
                            <span class="score-val ${out.is_knn ? 'bad' : ''}">${out.knn_similarity.toFixed(3)}</span>
                        </div>
                    </div>
                    
                    <div class="action-buttons-group" style="width: 100%;">
                        <button class="btn btn-danger btn-block btn-small btn-exclude-out" data-id="${out.id}">
                            <i class="fa-solid fa-trash"></i> Exclude / Delete
                        </button>
                        <button class="btn btn-secondary btn-block btn-small btn-keep-out" data-id="${out.id}">
                            <i class="fa-solid fa-check"></i> Keep Crop
                        </button>
                    </div>
                </div>
            `;
        });
        
        // Add card button event listeners
        gridContainer.querySelectorAll('.btn-exclude-out').forEach(btn => {
            btn.addEventListener('click', () => {
                excludeCrop(btn.getAttribute('data-id'), "Manually excluded class outlier crop.");
            });
        });
        
        gridContainer.querySelectorAll('.btn-keep-out').forEach(btn => {
            btn.addEventListener('click', () => {
                approveCrop(btn.getAttribute('data-id'), "Manually approved outlier crop.");
            });
        });
    }
    
    // --- 4. MISLABELS TAB ---
    function renderMislabelsTab() {
        const listContainer = document.getElementById('mis-list-container');
        const countTitle = document.getElementById('mis-items-title');
        
        const mislabels = reportData.mislabels || [];
        countTitle.textContent = `Suspected Wrong Labels (${mislabels.length})`;
        
        if (mislabels.length === 0) {
            listContainer.innerHTML = `
                <div class="empty-state-msg" style="height: 100px;">
                    <i class="fa-solid fa-tag" style="font-size:24px;"></i>
                    <p>No suspected mislabels found at current consensus threshold.</p>
                </div>
            `;
            document.getElementById('mis-empty-state').classList.remove('hidden');
            document.getElementById('mis-inspector-content').classList.add('hidden');
            return;
        }
        
        listContainer.innerHTML = '';
        mislabels.forEach((mis, idx) => {
            const isSelected = selectedIndex === mis.index;
            listContainer.innerHTML += `
                <div class="item-list-card ${isSelected ? 'active' : ''}" data-mis-idx="${idx}">
                    <div class="list-item-meta">
                        <div class="item-thumbnail">
                            <img src="${mis.path}" alt="${mis.id}">
                        </div>
                        <div class="item-text-info">
                            <span class="item-text-title">${mis.id}</span>
                            <span class="item-text-subtitle">Labeled: <b style="color:var(--color-outlier);">${mis.label}</b> &rarr; Suggest: <b style="color:var(--color-clean);">${mis.suggested_label}</b></span>
                        </div>
                    </div>
                    <div class="item-list-stat mis"><i class="fa-solid fa-triangle-exclamation"></i></div>
                </div>
            `;
        });
        
        // Add click handler
        listContainer.querySelectorAll('.item-list-card').forEach(card => {
            card.addEventListener('click', () => {
                const misIdx = parseInt(card.getAttribute('data-mis-idx'));
                selectMislabelItem(misIdx);
            });
        });
        
        // Manage active panel state
        const activeCard = listContainer.querySelector('.item-list-card.active');
        if (!activeCard) {
            document.getElementById('mis-empty-state').classList.remove('hidden');
            document.getElementById('mis-inspector-content').classList.add('hidden');
        }
    }
    
    function selectMislabelItem(idx) {
        const mis = reportData.mislabels[idx];
        if (!mis) return;
        
        selectedIndex = mis.index;
        
        // Highlight in list
        document.querySelectorAll('#mis-list-container .item-list-card').forEach((card, cidx) => {
            if (cidx === idx) card.classList.add('active');
            else card.classList.remove('active');
        });
        
        document.getElementById('mis-empty-state').classList.add('hidden');
        const inspector = document.getElementById('mis-inspector-content');
        inspector.classList.remove('hidden');
        
        // Bind UI values
        document.getElementById('mis-label-current').textContent = mis.label;
        document.getElementById('mis-label-suggested').textContent = mis.suggested_label;
        document.getElementById('mis-img-main').src = mis.path;
        document.getElementById('mis-id-main').textContent = mis.id;
        document.getElementById('mis-reason-text').textContent = mis.reason;
        
        document.getElementById('btn-relabel-text').textContent = mis.suggested_label;
        
        // Fill Nearest Neighbors list
        const neighborsList = document.getElementById('mis-neighbors-list');
        neighborsList.innerHTML = '';
        
        mis.neighbors.forEach(n => {
            const classColor = classColors[n.label] || defaultColor;
            neighborsList.innerHTML += `
                <div class="neighbor-card">
                    <div class="neighbor-meta">
                        <div class="neighbor-thumb">
                            <img src="${n.path}" alt="${n.id}">
                        </div>
                        <span class="neighbor-id">${n.id}</span>
                        <span class="class-pill neighbor-class" style="background-color: ${classColor}22; color: ${classColor}; border: 1px solid ${classColor}33;">${n.label}</span>
                    </div>
                    <span class="neighbor-sim">${(n.sim * 100).toFixed(1)}%</span>
                </div>
            `;
        });
        
        // Wire up buttons
        document.getElementById('btn-mis-relabel').onclick = () => relabelCrop(mis.id, mis.label, mis.suggested_label);
        document.getElementById('btn-mis-keep').onclick = () => approveCrop(mis.id, `Approved label '${mis.label}' after neighborhood check.`);
        document.getElementById('btn-mis-delete').onclick = () => excludeCrop(mis.id, "Deleted suspected mislabeled crop.");
    }
    
    // --- 5. EXPLORER TAB ---
    function renderExplorerTab() {
        const gridContainer = document.getElementById('explorer-crops-grid');
        const searchVal = document.getElementById('explorer-search').value.toLowerCase();
        const selectedClass = document.getElementById('filter-class').value;
        const selectedAnomaly = document.getElementById('filter-anomaly').value;
        const sortBy = document.getElementById('sort-by').value;
        
        const items = reportData.items;
        
        // A. Filter Items
        let filtered = items.map((item, idx) => ({ ...item, originalIdx: idx }));
        
        if (searchVal) {
            filtered = filtered.filter(item => item.id.toLowerCase().includes(searchVal));
        }
        
        if (selectedClass !== 'all') {
            filtered = filtered.filter(item => item.curLabel === selectedClass);
        }
        
        if (selectedAnomaly !== 'all') {
            if (selectedAnomaly === 'clean') {
                filtered = filtered.filter(item => !item.curDeleted && !item.is_duplicate && !item.is_outlier && !item.is_mislabeled);
            } else if (selectedAnomaly === 'flagged') {
                filtered = filtered.filter(item => !item.curDeleted && (item.is_duplicate || item.is_outlier || item.is_mislabeled));
            } else if (selectedAnomaly === 'duplicate') {
                filtered = filtered.filter(item => !item.curDeleted && item.is_duplicate);
            } else if (selectedAnomaly === 'outlier') {
                filtered = filtered.filter(item => !item.curDeleted && item.is_outlier);
            } else if (selectedAnomaly === 'mislabel') {
                filtered = filtered.filter(item => !item.curDeleted && item.is_mislabeled);
            } else if (selectedAnomaly === 'resolved') {
                filtered = filtered.filter(item => item.curDeleted || item.curKept || relabeledLabels.has(item.id));
            }
        }
        
        // B. Sort Items
        if (sortBy === 'index-asc') {
            filtered.sort((a, b) => a.originalIdx - b.originalIdx);
        } else if (sortBy === 'centroid-desc') {
            filtered.sort((a, b) => b.centroid_similarity - a.centroid_similarity);
        } else if (sortBy === 'centroid-asc') {
            filtered.sort((a, b) => a.centroid_similarity - b.centroid_similarity);
        }
        
        document.getElementById('explorer-grid-title').textContent = `Dataset Items (${filtered.length})`;
        
        // C. Render Grid
        gridContainer.innerHTML = '';
        if (filtered.length === 0) {
            gridContainer.innerHTML = `<p style="grid-column: 1/-1; color: var(--text-secondary); text-align: center; padding: 20px;">No items match the current filters.</p>`;
            return;
        }
        
        filtered.forEach(item => {
            let indicatorClass = "";
            if (item.curDeleted) indicatorClass = ""; // Excluded is just opacity
            else if (item.curKept) indicatorClass = "res";
            else if (item.is_duplicate) indicatorClass = "dup";
            else if (item.is_outlier) indicatorClass = "out";
            else if (item.is_mislabeled) indicatorClass = "mis";
            
            const isSelected = selectedIndex === item.originalIdx;
            
            gridContainer.innerHTML += `
                <div class="explorer-crop-thumb ${item.curDeleted ? 'resolved' : ''} ${isSelected ? 'active' : ''}" data-idx="${item.originalIdx}" title="${item.id} (${item.curLabel})">
                    <img src="${item.path}" alt="${item.id}">
                    ${indicatorClass ? `<div class="thumb-indicator ${indicatorClass}"></div>` : ''}
                </div>
            `;
        });
        
        // Wire up selection click
        gridContainer.querySelectorAll('.explorer-crop-thumb').forEach(thumb => {
            thumb.addEventListener('click', () => {
                const idx = parseInt(thumb.getAttribute('data-idx'));
                selectExplorerItem(idx);
            });
        });
        
        // Keep detailed view in sync if selected item changed
        if (selectedIndex >= 0) {
            showExplorerDetail(selectedIndex);
        } else {
            document.getElementById('exp-empty-state').classList.remove('hidden');
            document.getElementById('exp-inspector-content').classList.add('hidden');
        }
    }
    
    function selectExplorerItem(idx) {
        selectedIndex = idx;
        
        // Toggle active border state in DOM
        document.querySelectorAll('.explorer-crop-thumb').forEach(thumb => {
            if (parseInt(thumb.getAttribute('data-idx')) === idx) thumb.classList.add('active');
            else thumb.classList.remove('active');
        });
        
        showExplorerDetail(idx);
    }
    
    function showExplorerDetail(idx) {
        const item = reportData.items[idx];
        if (!item) return;
        
        document.getElementById('exp-empty-state').classList.add('hidden');
        const content = document.getElementById('exp-inspector-content');
        content.classList.remove('hidden');
        
        // Bind UI Elements
        document.getElementById('exp-img').src = item.path;
        document.getElementById('exp-id').textContent = item.id;
        
        const labelPill = document.getElementById('exp-label');
        labelPill.textContent = item.curLabel;
        const color = classColors[item.curLabel] || defaultColor;
        labelPill.style.backgroundColor = color + "22";
        labelPill.style.color = color;
        labelPill.style.border = `1px solid ${color}33`;
        
        document.getElementById('exp-sim-centroid').textContent = item.centroid_similarity !== undefined ? item.centroid_similarity.toFixed(4) : "N/A";
        document.getElementById('exp-sim-knn').textContent = item.knn_similarity !== undefined ? item.knn_similarity.toFixed(4) : "N/A";
        
        // Set glow border based on quality state
        const glowBorder = document.getElementById('exp-glow-border');
        glowBorder.className = "crop-image-wrapper border-glow";
        if (item.is_duplicate) glowBorder.classList.add('dup');
        else if (item.is_outlier) glowBorder.classList.add('out');
        else if (item.is_mislabeled) glowBorder.classList.add('mis');
        
        // Fill Status badges
        const badgesContainer = document.getElementById('exp-status-badges');
        badgesContainer.innerHTML = '';
        if (item.curDeleted) {
            badgesContainer.innerHTML += `<span class="exp-badge out">Deleted / Excluded</span>`;
        } else if (item.curKept) {
            badgesContainer.innerHTML += `<span class="exp-badge res">Approved / Kept</span>`;
        } else {
            let isClean = true;
            if (item.is_duplicate) {
                badgesContainer.innerHTML += `<span class="exp-badge dup">Duplicate</span>`;
                isClean = false;
            }
            if (item.is_outlier) {
                badgesContainer.innerHTML += `<span class="exp-badge out">Outlier</span>`;
                isClean = false;
            }
            if (item.is_mislabeled) {
                badgesContainer.innerHTML += `<span class="exp-badge mis">Suspected Mislabel</span>`;
                isClean = false;
            }
            if (isClean) {
                badgesContainer.innerHTML += `<span class="exp-badge clean">Clean</span>`;
            }
        }
        
        // Manage active actions state based on item status
        const btnKeep = document.getElementById('btn-exp-keep');
        const btnRelabel = document.getElementById('btn-exp-relabel');
        const btnDelete = document.getElementById('btn-exp-delete');
        
        if (item.curDeleted) {
            btnKeep.textContent = "Restore Crop";
            btnKeep.className = "btn btn-success btn-full";
            btnKeep.onclick = () => restoreCrop(item.id);
            btnRelabel.disabled = true;
            btnDelete.disabled = true;
        } else {
            btnKeep.textContent = "Keep / Approve Crop";
            btnKeep.className = "btn btn-secondary btn-full";
            btnKeep.disabled = item.curKept || (!item.is_duplicate && !item.is_outlier && !item.is_mislabeled);
            btnKeep.onclick = () => approveCrop(item.id, "Approved during explorer review.");
            
            btnRelabel.disabled = false;
            btnDelete.disabled = false;
            btnDelete.onclick = () => excludeCrop(item.id, "Deleted during explorer review.");
        }
        
        // Hide relabel panel initially
        document.getElementById('relabel-dropdown-container').classList.add('hidden');
        btnRelabel.onclick = () => {
            const dropdown = document.getElementById('relabel-dropdown-container');
            dropdown.classList.toggle('hidden');
            document.getElementById('exp-relabel-select').value = item.curLabel;
        };
    }
    
    // ==========================================================================
    // ANOMALY OVERRIDES / HUMAN ACTION REGISTER
    // ==========================================================================
    
    function excludeCrop(id, description = "") {
        deletedIds.add(id);
        registerAction({
            type: 'exclude_crop',
            id: id,
            description: description || `Excluded crop ${id} from dataset.`
        });
        
        // Reset current selection detail if it was deleted
        recomputeCuration();
    }
    
    function restoreCrop(id) {
        deletedIds.delete(id);
        
        // Remove from actions log
        const idx = actionsLog.findIndex(act => act.type === 'exclude_crop' && act.id === id);
        if (idx >= 0) actionsLog.splice(idx, 1);
        
        registerAction({
            type: 'restore_crop',
            id: id,
            description: `Restored crop ${id} back to active dataset.`
        });
        
        recomputeCuration();
    }
    
    function approveCrop(id, description = "") {
        keptIds.add(id);
        registerAction({
            type: 'approve_crop',
            id: id,
            description: description || `Approved crop ${id} as valid.`
        });
        recomputeCuration();
    }
    
    function relabelCrop(id, oldLabel, newLabel) {
        relabeledLabels.set(id, newLabel);
        registerAction({
            type: 'relabel_crop',
            id: id,
            old_label: oldLabel,
            new_label: newLabel,
            description: `Relabeled crop ${id} from '${oldLabel}' to '${newLabel}'.`
        });
        recomputeCuration();
    }
    
    function registerAction(act) {
        // Prevent duplicate logs for the same crop and type
        const existingIdx = actionsLog.findIndex(a => a.id === act.id && a.type === act.type);
        if (existingIdx >= 0) {
            actionsLog[existingIdx] = act;
        } else {
            actionsLog.push(act);
        }
    }
    
    function updateExportBar() {
        const bar = document.querySelector('.export-floating-bar');
        const summaryText = document.getElementById('export-actions-summary');
        
        if (actionsLog.length > 0) {
            bar.style.display = 'block';
            summaryText.textContent = `${actionsLog.length} curation action(s) ready to export`;
        } else {
            bar.style.display = 'none';
        }
    }
    
    // ==========================================================================
    // INTERACTIVE CANVAS PLOT RENDERING
    // ==========================================================================
    
    function getProjectedCoord(idx) {
        const coords = pcaCoordinates[idx];
        if (!coords) return [0, 0, 0];
        const x = coords[0];
        const y = coords[1];
        const z = coords[2] || 0;
        
        if (!is3D) {
            return [x, y, 0];
        }
        
        const cosY = Math.cos(rotationY);
        const sinY = Math.sin(rotationY);
        const cosX = Math.cos(rotationX);
        const sinX = Math.sin(rotationX);

        // Rotate Y
        const x1 = x * cosY - z * sinY;
        const z1 = x * sinY + z * cosY;

        // Rotate X
        const y2 = y * cosX - z1 * sinX;
        const z2 = y * sinX + z1 * cosX;
        
        return [x1, y2, z2];
    }
    
    function resetCanvasView() {
        if (pcaCoordinates.length === 0) return;
        
        // Find dimensions min/max
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
        pcaCoordinates.forEach((_, i) => {
            const [x, y] = getProjectedCoord(i);
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        });
        
        const padX = (maxX - minX) * 0.15 || 1.0;
        const padY = (maxY - minY) * 0.15 || 1.0;
        
        minX -= padX; maxX += padX;
        minY -= padY; maxY += padY;
        
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        
        // Scale to fit canvas
        const scaleX = width / (maxX - minX);
        const scaleY = height / (maxY - minY);
        viewZoom = Math.min(scaleX, scaleY);
        
        // Pan to center
        viewPanX = width / 2 - (minX + maxX) / 2 * viewZoom;
        viewPanY = height / 2 - (minY + maxY) / 2 * viewZoom;
        
        drawCanvas();
    }
    
    function drawCanvas() {
        if (!reportData || pcaCoordinates.length === 0) return;
        
        const N = pcaCoordinates.length;
        
        // Adjust Canvas size for high-DPI displays
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        
        const width = rect.width;
        const height = rect.height;
        
        // Clear background
        ctx.fillStyle = '#090c10';
        ctx.fillRect(0, 0, width, height);
        
        // Draw grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        ctx.lineWidth = 1;
        const gridSize = 40 * viewZoom;
        
        const startX = viewPanX % gridSize;
        for (let x = startX; x < width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        
        const startY = viewPanY % gridSize;
        for (let y = startY; y < height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        const items = reportData.items;
        
        // Calculate all projected 2D coordinates
        const projectedCoords = pcaCoordinates.map((_, i) => getProjectedCoord(i));
        
        // Compute minZ/maxZ for depth cues
        let minZ = Infinity, maxZ = -Infinity;
        if (is3D) {
            projectedCoords.forEach(coord => {
                const z = coord[2];
                if (z < minZ) minZ = z;
                if (z > maxZ) maxZ = z;
            });
        }
        
        // Determine rendering order (depth sorting)
        const renderOrder = [];
        for (let i = 0; i < N; i++) renderOrder.push(i);
        if (is3D) {
            renderOrder.sort((a, b) => projectedCoords[a][2] - projectedCoords[b][2]);
        }
        
        // First Draw: Anomaly Rings/Halos (Bottom Layer)
        if (showAnomalyRings) {
            renderOrder.forEach(i => {
                const item = items[i];
                if (item.curDeleted) return; // Hide rings for deleted
                
                const [px, py, pz] = projectedCoords[i];
                
                // Screen projection coordinates
                const sx = px * viewZoom + viewPanX;
                const sy = py * viewZoom + viewPanY;
                
                if (sx < -20 || sx > width + 20 || sy < -20 || sy > height + 20) return;
                
                let depthFactor = 1.0;
                if (is3D && maxZ > minZ) {
                    depthFactor = 0.4 + 0.8 * ((pz - minZ) / (maxZ - minZ));
                }
                
                ctx.save();
                if (is3D) {
                    ctx.globalAlpha = Math.max(0.1, depthFactor * 0.8);
                }
                
                if (item.is_duplicate) {
                    drawGlowingHalo(sx, sy, 14 * depthFactor, 'rgba(0, 242, 254, 0.25)', 'rgba(0, 242, 254, 0)');
                } else if (item.is_outlier) {
                    drawGlowingHalo(sx, sy, 14 * depthFactor, 'rgba(244, 63, 94, 0.25)', 'rgba(244, 63, 94, 0)');
                } else if (item.is_mislabeled) {
                    drawGlowingHalo(sx, sy, 14 * depthFactor, 'rgba(245, 158, 11, 0.25)', 'rgba(245, 158, 11, 0)');
                }
                
                ctx.restore();
            });
        }
        
        // Second Draw: Points Core
        renderOrder.forEach(i => {
            const item = items[i];
            const [px, py, pz] = projectedCoords[i];
            const sx = px * viewZoom + viewPanX;
            const sy = py * viewZoom + viewPanY;
            
            if (sx < -10 || sx > width + 10 || sy < -10 || sy > height + 10) return;
            
            const color = classColors[item.curLabel] || defaultColor;
            
            let depthFactor = 1.0;
            if (is3D && maxZ > minZ) {
                depthFactor = 0.4 + 0.8 * ((pz - minZ) / (maxZ - minZ));
            }
            
            ctx.save();
            if (is3D && !item.curDeleted) {
                ctx.globalAlpha = Math.max(0.15, depthFactor);
            }
            
            ctx.beginPath();
            
            if (item.curDeleted) {
                // Render excluded items as small faded gray crosses
                ctx.strokeStyle = '#444';
                ctx.lineWidth = 1.5;
                const size = 3 * depthFactor;
                ctx.moveTo(sx - size, sy - size);
                ctx.lineTo(sx + size, sy + size);
                ctx.moveTo(sx + size, sy - size);
                ctx.lineTo(sx - size, sy + size);
                ctx.stroke();
            } else {
                // Active points
                const isSelected = selectedIndex === i;
                const isHovered = hoveredIndex === i;
                
                let radius = 5 * depthFactor;
                if (isHovered) radius = 8 * depthFactor;
                if (isSelected) radius = 9 * depthFactor;
                
                ctx.arc(sx, sy, radius, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                
                // Active outline borders
                if (isSelected) {
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 2.5;
                    ctx.stroke();
                    
                    // Outer accent ring
                    ctx.beginPath();
                    ctx.arc(sx, sy, radius + 4 * depthFactor, 0, Math.PI * 2);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                } else if (isHovered) {
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                } else {
                    ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
            
            ctx.restore();
        });
        
        // Third Draw: Tooltip Overlay (Top Layer)
        if (hoveredIndex >= 0 && hoveredIndex < N) {
            const item = items[hoveredIndex];
            if (!item.curDeleted) {
                const [px, py] = projectedCoords[hoveredIndex];
                const sx = px * viewZoom + viewPanX;
                const sy = py * viewZoom + viewPanY;
                drawTooltip(sx, sy, item);
            }
        }
    }
    
    function drawGlowingHalo(x, y, r, innerColor, outerColor) {
        const grad = ctx.createRadialGradient(x, y, 4, x, y, r);
        grad.addColorStop(0, innerColor);
        grad.addColorStop(1, outerColor);
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
    }
    
    function drawTooltip(x, y, item) {
        ctx.save();
        
        const titleText = `${item.id}`;
        const labelText = `Label: ${item.curLabel}`;
        
        let tags = [];
        if (item.is_duplicate) tags.push("DUPLICATE");
        if (item.is_outlier) tags.push("OUTLIER");
        if (item.is_mislabeled) tags.push("MISLABELED");
        
        // Calculate dynamic dimensions
        ctx.font = 'bold 12px var(--font-sans)';
        const w1 = ctx.measureText(titleText).width;
        ctx.font = '11px var(--font-sans)';
        const w2 = ctx.measureText(labelText).width;
        
        const boxWidth = Math.max(140, w1 + 30, w2 + 20);
        const boxHeight = tags.length > 0 ? 64 : 46;
        
        // Align tooltip above point
        const tx = Math.max(10, Math.min(canvas.clientWidth - boxWidth - 10, x - boxWidth / 2));
        const ty = y - boxHeight - 12;
        
        // Draw background card with blur border
        ctx.fillStyle = 'rgba(13, 17, 23, 0.95)';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 1;
        
        // Rounded Rect
        const r = 6;
        ctx.beginPath();
        ctx.moveTo(tx + r, ty);
        ctx.lineTo(tx + boxWidth - r, ty);
        ctx.quadraticCurveTo(tx + boxWidth, ty, tx + boxWidth, ty + r);
        ctx.lineTo(tx + boxWidth, ty + boxHeight - r);
        ctx.quadraticCurveTo(tx + boxWidth, ty + boxHeight, tx + boxWidth - r, ty + boxHeight);
        ctx.lineTo(tx + r, ty + boxHeight);
        ctx.quadraticCurveTo(tx, ty + boxHeight, tx, ty + boxHeight - r);
        ctx.lineTo(tx, ty + r);
        ctx.quadraticCurveTo(tx, ty, tx + r, ty);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        
        // Text Strings
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px var(--font-sans)';
        ctx.fillText(titleText, tx + 10, ty + 18);
        
        ctx.fillStyle = 'var(--text-secondary)';
        ctx.font = '11px var(--font-sans)';
        ctx.fillText(labelText, tx + 10, ty + 32);
        
        // Draw sub-indicators if anomalies exist
        if (tags.length > 0) {
            let txOffset = tx + 10;
            tags.forEach(tag => {
                let badgeColor = 'var(--text-secondary)';
                if (tag === 'DUPLICATE') badgeColor = 'var(--color-duplicate)';
                else if (tag === 'OUTLIER') badgeColor = 'var(--color-outlier)';
                else if (tag === 'MISLABELED') badgeColor = 'var(--color-mislabel)';
                
                ctx.fillStyle = badgeColor;
                ctx.font = 'bold 9px var(--font-sans)';
                const tagW = ctx.measureText(tag).width;
                
                ctx.fillRect(txOffset - 2, ty + 39, tagW + 4, 14);
                ctx.fillStyle = '#090c10';
                ctx.fillText(tag, txOffset, ty + 50);
                txOffset += tagW + 10;
            });
        }
        
        ctx.restore();
    }
    
    // ==========================================================================
    // INTERACTION CONTROLLERS (Sliders, Canvas, Upload, Export)
    // ==========================================================================
    
    function setupSliders() {
        const sliderDup = document.getElementById('slider-dup');
        const sliderOutCentroid = document.getElementById('slider-out-centroid');
        const sliderMisK = document.getElementById('slider-mis-k');
        const sliderMisConsensus = document.getElementById('slider-mis-consensus');
        
        sliderDup.addEventListener('input', (e) => {
            dupThreshold = parseFloat(e.target.value);
            document.getElementById('val-dup').textContent = dupThreshold.toFixed(2);
            recomputeCuration();
        });
        
        sliderOutCentroid.addEventListener('input', (e) => {
            outCentroidThreshold = parseFloat(e.target.value);
            document.getElementById('val-out-centroid').textContent = outCentroidThreshold.toFixed(2);
            recomputeCuration();
        });
        
        sliderMisK.addEventListener('input', (e) => {
            mislabelK = parseInt(e.target.value);
            document.getElementById('val-mis-k').textContent = mislabelK;
            recomputeCuration();
        });
        
        sliderMisConsensus.addEventListener('input', (e) => {
            mislabelConsensus = parseFloat(e.target.value);
            document.getElementById('val-mis-consensus').textContent = mislabelConsensus.toFixed(2);
            recomputeCuration();
        });
    }
    
    function setupCanvas() {
        // Drag to Pan or Rotate
        canvas.addEventListener('mousedown', (e) => {
            isDrawing = true;
            dragStart.x = e.clientX;
            dragStart.y = e.clientY;
            dragPanStart.x = viewPanX;
            dragPanStart.y = viewPanY;
            dragRotationStart.x = rotationX;
            dragRotationStart.y = rotationY;
        });
        
        window.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            if (isDrawing) {
                const dx = e.clientX - dragStart.x;
                const dy = e.clientY - dragStart.y;
                if (is3D) {
                    // Update 3D rotation based on mouse movement
                    rotationY = dragRotationStart.y + dx * 0.007;
                    rotationX = dragRotationStart.x + dy * 0.007;
                } else {
                    // Calculate Panned offset
                    viewPanX = dragPanStart.x + dx;
                    viewPanY = dragPanStart.y + dy;
                }
                drawCanvas();
            } else {
                // Hover detection
                if (pcaCoordinates.length === 0) return;
                
                let prevHovered = hoveredIndex;
                hoveredIndex = -1;
                
                let minDist = 10; // Click radius
                
                pcaCoordinates.forEach((_, idx) => {
                    const item = reportData.items[idx];
                    if (item.curDeleted) return; // Ignore hovered excluded
                    
                    const [px, py] = getProjectedCoord(idx);
                    const sx = px * viewZoom + viewPanX;
                    const sy = py * viewZoom + viewPanY;
                    
                    const dx = mouseX - sx;
                    const dy = mouseY - sy;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    
                    if (dist < minDist) {
                        minDist = dist;
                        hoveredIndex = idx;
                    }
                });
                
                if (hoveredIndex !== prevHovered) {
                    drawCanvas();
                }
            }
        });
        
        window.addEventListener('mouseup', () => {
            isDrawing = false;
        });
        
        // Scroll Wheel to Zoom
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            // Calculate mouse position relative to data space before zoom
            const [dataX, dataY] = is3D ? [0, 0] : [
                (mouseX - viewPanX) / viewZoom,
                (mouseY - viewPanY) / viewZoom
            ];
            
            const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
            viewZoom = Math.max(0.1, Math.min(50, viewZoom * zoomFactor));
            
            if (!is3D) {
                // Reposition pan so mouse point remains static
                viewPanX = mouseX - dataX * viewZoom;
                viewPanY = mouseY - dataY * viewZoom;
            }
            
            drawCanvas();
        }, { passive: false });
        
        // Canvas Toolbar Controls
        document.getElementById('btn-plot-reset').addEventListener('click', () => {
            if (is3D) {
                rotationX = -0.5;
                rotationY = 0.6;
            }
            resetCanvasView();
        });
        
        document.getElementById('btn-plot-toggle-flags').addEventListener('click', (e) => {
            showAnomalyRings = !showAnomalyRings;
            e.target.classList.toggle('active', showAnomalyRings);
            drawCanvas();
        });
        
        // 3D Mode Toggle Listener
        document.getElementById('btn-plot-toggle-3d').addEventListener('click', (e) => {
            is3D = !is3D;
            e.target.classList.toggle('active', is3D);
            
            const cardTitle = document.getElementById('plot-card-title');
            const instructions = document.getElementById('plot-instructions');
            
            if (is3D) {
                cardTitle.innerHTML = '<i class="fa-solid fa-cube"></i> Interactive 3D Embedding Space (PCA Projection)';
                instructions.innerHTML = '<i class="fa-solid fa-arrows-spin"></i> Drag to rotate | <i class="fa-solid fa-magnifying-glass-plus"></i> Scroll to zoom | Click point to inspect';
            } else {
                cardTitle.innerHTML = '<i class="fa-solid fa-circle-nodes"></i> Interactive 2D Embedding Space (PCA Projection)';
                instructions.innerHTML = '<i class="fa-solid fa-arrows-up-down-left-right"></i> Drag to pan | <i class="fa-solid fa-magnifying-glass-plus"></i> Scroll to zoom | Click point to inspect';
            }
            
            resetCanvasView();
        });
        
        // Click to Select item
        canvas.addEventListener('click', (e) => {
            if (isDrawing) {
                const clickMoveDist = Math.sqrt(Math.pow(e.clientX - dragStart.x, 2) + Math.pow(e.clientY - dragStart.y, 2));
                if (clickMoveDist > 3) return; // It was a drag rotation/pan
            }
            
            if (hoveredIndex >= 0) {
                // Open Item in Explorer
                switchTab('explorer');
                selectExplorerItem(hoveredIndex);
            }
        });
    }
    
    function setupActionButtons() {
        // Overview link cards
        document.querySelectorAll('.stat-card.clickable').forEach(card => {
            card.addEventListener('click', () => {
                switchTab(card.getAttribute('data-tab-link'));
            });
        });
        
        // Explorer Panel Action Overrides
        const btnExpConfirmRelabel = document.getElementById('btn-exp-confirm-relabel');
        const btnExpCancelRelabel = document.getElementById('btn-exp-cancel-relabel');
        
        btnExpConfirmRelabel.onclick = () => {
            const item = reportData.items[selectedIndex];
            if (!item) return;
            const newLabel = document.getElementById('exp-relabel-select').value;
            relabelCrop(item.id, item.curLabel, newLabel);
            document.getElementById('relabel-dropdown-container').classList.add('hidden');
        };
        
        btnExpCancelRelabel.onclick = () => {
            document.getElementById('relabel-dropdown-container').classList.add('hidden');
        };
        
        // Export Action Button handler
        document.getElementById('btn-export-actions').addEventListener('click', exportCurationReport);
        document.getElementById('btn-commit-writeback').addEventListener('click', commitWriteback);
    }
    
    // --- FILE UPLOAD ENGINE ---
    function handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = function(evt) {
            try {
                const report = JSON.parse(evt.target.result);
                if (!report.items || !report.summary) {
                    alert("Invalid report format! Curation report must contain 'items' list and 'summary' details.");
                    return;
                }
                loadDataset(report);
                alert("Dataset curation report successfully loaded!");
            } catch(err) {
                alert("Error parsing file! Please upload a valid JSON curation report: " + err.message);
            }
        };
        reader.readAsText(file);
    }
    
    // --- FILE EXPORT ENGINE (Human-in-the-loop audit file) ---
    function exportCurationReport() {
        if (!reportData) return;
        
        // Construct final curated items list
        const curatedItems = reportData.items.map(item => {
            const hasRelabel = relabeledLabels.has(item.id);
            const hasDelete = deletedIds.has(item.id);
            
            return {
                id: item.id,
                original_label: item.label,
                curated_label: hasDelete ? null : (hasRelabel ? relabeledLabels.get(item.id) : item.label),
                status: hasDelete ? "excluded" : (hasRelabel ? "relabeled" : "retained"),
                is_outlier: item.is_outlier,
                is_duplicate: item.is_duplicate,
                is_mislabeled: item.is_mislabeled
            };
        });
        
        const auditLog = {
            export_timestamp: new Date().toISOString(),
            dataset_summary: {
                original_samples: reportData.items.length,
                curated_samples: reportData.items.length - deletedIds.size,
                excluded_count: deletedIds.size,
                relabeled_count: relabeledLabels.size,
                approved_anomaly_overrides: keptIds.size
            },
            actions_executed: actionsLog,
            curated_dataset: curatedItems
        };
        
        // Download File trigger
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(auditLog, null, 2));
        const dlAnchorElem = document.createElement('a');
        dlAnchorElem.setAttribute("href",     dataStr     );
        dlAnchorElem.setAttribute("download", `curated_dataset_actions_${Date.now()}.json`);
        dlAnchorElem.click();
        
        // Clear log state after successful export
        alert("Curation log exported successfully!");
    }
    
    function commitWriteback() {
        if (!reportData) return;
        
        if (deletedIds.size === 0 && relabeledLabels.size === 0) {
            alert("No actions to commit! Please exclude or relabel some crops first.");
            return;
        }
        
        if (!confirm(`Are you sure you want to commit ${deletedIds.size} deletion(s) and ${relabeledLabels.size} relabeling(s) directly to the dataset files? This will overwrite your raw YOLO label files.`)) {
            return;
        }
        
        const actions = [];
        
        // Excluded crops
        deletedIds.forEach(id => {
            const item = reportData.items.find(it => it.id === id);
            if (item && item.original_image_path) {
                actions.push({
                    original_image_path: item.original_image_path,
                    crop_idx: item.crop_idx,
                    bbox_norm: item.bbox_norm,
                    action: 'DELETE'
                });
            }
        });
        
        // Relabeled crops
        relabeledLabels.forEach((newLabel, id) => {
            const item = reportData.items.find(it => it.id === id);
            if (item && item.original_image_path) {
                actions.push({
                    original_image_path: item.original_image_path,
                    crop_idx: item.crop_idx,
                    bbox_norm: item.bbox_norm,
                    action: 'RELABEL',
                    new_class_id: newLabel
                });
            }
        });
        
        // Make POST request to /api/writeback
        fetch('/api/writeback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ actions: actions })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server returned status ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'success') {
                alert(`Success! ${data.message}`);
                // Clear the curation state after successful writeback
                deletedIds.clear();
                relabeledLabels.clear();
                actionsLog.length = 0;
                recomputeCuration();
            } else {
                alert(`Error during writeback: ${data.message}`);
            }
        })
        .catch(error => {
            console.error('Writeback failed:', error);
            alert(`Writeback failed! Make sure the CleanCrop server is running and try again. Error: ${error.message}`);
        });
    }
    
    function showErrorState(msg) {
        document.querySelector('.content-scroll').innerHTML = `
            <div class="empty-state-msg" style="height: 400px; grid-column: 1/-1;">
                <i class="fa-solid fa-triangle-exclamation" style="font-size:48px; color:var(--color-outlier);"></i>
                <h2>Curation Environment Error</h2>
                <p>${msg}</p>
            </div>
        `;
    }
    
    // --- EXPLORER FILTER TRIGGERS ---
    document.getElementById('explorer-search').addEventListener('input', renderExplorerTab);
    document.getElementById('filter-class').addEventListener('change', renderExplorerTab);
    document.getElementById('filter-anomaly').addEventListener('change', renderExplorerTab);
    document.getElementById('sort-by').addEventListener('change', renderExplorerTab);

})();
