/**
 * CleanCrop box editor + rare-class gallery (loaded after app.js).
 *  - Box editor: edit every box of one image in the browser (select, move, resize, change class, delete,
 *    draw new boxes). Save writes the image's YOLO label file through /api/editimage (backup kept).
 *    Open it from the full-image view ("Edit boxes" button or key E) or from the rare-class gallery.
 *  - Rare classes: sidebar page showing every box of the less common classes at once, one class per tab.
 * Keys in the editor: drag on empty area = new box, Del = delete box, 1-9 = class of the selected box,
 * Tab = next box, wheel = zoom, right-drag = pan, 0 = fit, Ctrl+S = save, Esc = close.
 */
(function () {
    const PALETTE = ['#38bdf8', '#f472b6', '#facc15', '#a78bfa', '#34d399', '#fb923c'];
    const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const css = `
    .cc-overlay{position:fixed;inset:0;z-index:2000;background:rgba(5,8,12,.96);display:flex;flex-direction:column;font-family:inherit;color:#c9d1d9}
    .cc-bar{display:flex;gap:10px;align-items:center;padding:10px 14px;border-bottom:1px solid rgba(240,246,252,.1);flex-wrap:wrap}
    .cc-bar b{color:#f0f6fc}.cc-bar .cc-hint{color:#8b949e;font-size:12px;flex:1;min-width:260px}
    .cc-bar select,.cc-bar button{background:#161b22;color:#c9d1d9;border:1px solid rgba(240,246,252,.2);border-radius:6px;padding:6px 10px;font-size:13px;cursor:pointer}
    .cc-bar button.cc-primary{background:#10b981;color:#04130d;border-color:#10b981;font-weight:600}
    .cc-bar button.cc-danger{border-color:#f43f5e;color:#f43f5e}
    .cc-stage{flex:1;position:relative;overflow:hidden}.cc-stage canvas{position:absolute;inset:0;cursor:crosshair}
    .cc-msg{position:absolute;right:14px;bottom:14px;background:#161b22;padding:8px 12px;border-radius:6px;font-size:13px}
    .cc-tabs{display:flex;gap:8px;padding:10px 14px;flex-wrap:wrap}.cc-tabs button.active{background:#38bdf8;color:#04121b;border-color:#38bdf8}
    .cc-grid{flex:1;overflow:auto;padding:14px;display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;align-content:start}
    .cc-card{background:rgba(22,27,34,.8);border:1px solid rgba(240,246,252,.1);border-radius:8px;padding:6px;cursor:pointer;font-size:11px;color:#8b949e;word-break:break-all}
    .cc-card:hover{border-color:#38bdf8}.cc-card img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:4px;display:block;margin-bottom:4px}
    .cc-editbtn{position:absolute;top:14px;right:14px;z-index:5;background:#10b981;color:#04130d;border:0;border-radius:6px;padding:8px 14px;font-weight:600;cursor:pointer}`;
    document.head.insertAdjacentHTML('beforeend', `<style>${css}</style>`);

    const report = () => window.CURATION_REPORT;
    const names = () => (report() && report().class_names) || [];
    const color = c => PALETTE[c % PALETTE.length];

    // ======================================================================== box editor
    const E = { open: false, id: null, img: null, boxes: [], sel: -1, cls: 0, dirty: false,
                zoom: 1, panX: 0, panY: 0, drag: null, el: null, cv: null, ctx: null };

    function buildEditor() {
        const el = document.createElement('div');
        el.className = 'cc-overlay'; el.style.display = 'none';
        el.innerHTML = `<div class="cc-bar">
            <b id="cc-title">Box editor</b>
            <label>Class <select id="cc-class"></select></label>
            <button class="cc-danger" id="cc-del">Delete box (Del)</button>
            <button id="cc-fit">Fit (0)</button>
            <span class="cc-hint">Drag on an empty area = new box · drag a box to move, its corners to resize · 1-9 = class ·
              Tab = next box · wheel = zoom · right-drag = pan</span>
            <button id="cc-cancel">Close (Esc)</button>
            <button class="cc-primary" id="cc-save">Save (Ctrl+S)</button></div>
            <div class="cc-stage"><canvas></canvas><div class="cc-msg" id="cc-msg"></div></div>`;
        document.body.appendChild(el);
        E.el = el; E.cv = el.querySelector('canvas'); E.ctx = E.cv.getContext('2d');
        el.querySelector('#cc-save').onclick = save;
        el.querySelector('#cc-cancel').onclick = close;
        el.querySelector('#cc-del').onclick = delSel;
        el.querySelector('#cc-fit').onclick = () => { fit(); draw(); };
        el.querySelector('#cc-class').onchange = e => {
            E.cls = +e.target.value;
            if (E.sel >= 0) { E.boxes[E.sel].c = E.cls; E.dirty = true; draw(); }
        };
        E.cv.addEventListener('contextmenu', e => e.preventDefault());
        E.cv.addEventListener('mousedown', down);
        window.addEventListener('mousemove', move);
        window.addEventListener('mouseup', up);
        E.cv.addEventListener('wheel', wheel, { passive: false });
        window.addEventListener('resize', () => { if (E.open) { fit(); draw(); } });
    }

    async function open(id) {
        if (!E.el) buildEditor();
        msg('Loading…');
        const res = await fetch(`/api/labels?id=${encodeURIComponent(id)}&t=${Date.now()}`);
        if (!res.ok) { alert('Could not load the labels of this image (is the CleanCrop server running?)'); return; }
        const lab = await res.json();
        const img = new Image();
        img.src = `/api/image?id=${encodeURIComponent(id)}&t=${Date.now()}`;
        await img.decode();
        Object.assign(E, { open: true, id, img, sel: -1, dirty: false,
            boxes: lab.boxes.map(b => ({ c: b.class_id, x1: b.bbox[0] - b.bbox[2] / 2, y1: b.bbox[1] - b.bbox[3] / 2,
                                         x2: b.bbox[0] + b.bbox[2] / 2, y2: b.bbox[1] + b.bbox[3] / 2 })) });
        const item = (report().items || []).find(it => it.id === id);
        E.cls = item ? item.class_id : 0;
        E.sel = item ? E.boxes.findIndex(b => b.c === item.class_id && Math.abs((b.x1 + b.x2) / 2 - item.bbox[0]) < 1e-4
                                                && Math.abs((b.y1 + b.y2) / 2 - item.bbox[1]) < 1e-4) : -1;
        E.el.querySelector('#cc-class').innerHTML = names().map((n, i) => `<option value="${i}">${i + 1}. ${esc(n)}</option>`).join('');
        E.el.querySelector('#cc-class').value = String(E.sel >= 0 ? E.boxes[E.sel].c : E.cls);
        E.el.querySelector('#cc-title').textContent = lab.image;
        E.el.style.display = 'flex';
        fit();
        if (E.sel >= 0) zoomTo(E.boxes[E.sel]);
        draw(); msg(`${E.boxes.length} box(es)`);
    }

    function close() {
        if (E.dirty && !confirm('Close without saving your box changes?')) return;
        E.open = false; E.el.style.display = 'none';
    }

    async function save() {
        const boxes = E.boxes.filter(b => b.x2 - b.x1 > 1e-4 && b.y2 - b.y1 > 1e-4).map(b => {
            const x1 = Math.max(0, Math.min(b.x1, b.x2)), x2 = Math.min(1, Math.max(b.x1, b.x2));
            const y1 = Math.max(0, Math.min(b.y1, b.y2)), y2 = Math.min(1, Math.max(b.y1, b.y2));
            return { class_id: b.c, bbox: [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1] };
        });
        msg('Saving…');
        const res = await fetch('/api/editimage', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                    body: JSON.stringify({ id: E.id, boxes }) });
        const out = await res.json().catch(() => ({ status: 'error', message: res.statusText }));
        if (out.status !== 'success') { alert('Save failed: ' + (out.message || res.status)); msg('Save failed'); return; }
        E.dirty = false;
        msg('Saved. Reloading the dashboard…');
        sessionStorage.setItem('cc-after-edit', out.message);
        setTimeout(() => location.reload(), 600);
    }

    // view: image coords (0..1) <-> canvas px
    function fit() {
        const r = E.cv.parentElement.getBoundingClientRect();
        E.cv.width = r.width; E.cv.height = r.height;
        E.zoom = Math.min(r.width / E.img.width, r.height / E.img.height);
        E.panX = (r.width - E.img.width * E.zoom) / 2; E.panY = (r.height - E.img.height * E.zoom) / 2;
    }
    function zoomTo(b) {  // centre and enlarge a small box so it is easy to edit
        const bw = (b.x2 - b.x1) * E.img.width, bh = (b.y2 - b.y1) * E.img.height;
        const z = Math.min(E.zoom * 4, Math.min(E.cv.width, E.cv.height) / (6 * Math.max(bw, bh, 8)));
        if (z <= E.zoom) return;
        E.zoom = z;
        E.panX = E.cv.width / 2 - (b.x1 + b.x2) / 2 * E.img.width * z;
        E.panY = E.cv.height / 2 - (b.y1 + b.y2) / 2 * E.img.height * z;
    }
    const toPx = (x, y) => [E.panX + x * E.img.width * E.zoom, E.panY + y * E.img.height * E.zoom];
    const toImg = (px, py) => [(px - E.panX) / (E.img.width * E.zoom), (py - E.panY) / (E.img.height * E.zoom)];

    function draw() {
        const c = E.ctx;
        c.clearRect(0, 0, E.cv.width, E.cv.height);
        c.imageSmoothingEnabled = E.zoom < 2;
        c.drawImage(E.img, E.panX, E.panY, E.img.width * E.zoom, E.img.height * E.zoom);
        E.boxes.forEach((b, i) => {
            const [x1, y1] = toPx(b.x1, b.y1), [x2, y2] = toPx(b.x2, b.y2);
            c.strokeStyle = color(b.c); c.lineWidth = i === E.sel ? 3 : 1.5;
            c.strokeRect(x1, y1, x2 - x1, y2 - y1);
            c.font = '12px sans-serif'; c.fillStyle = color(b.c);
            c.fillText(names()[b.c] ?? b.c, x1 + 2, Math.max(12, y1 - 4));
            if (i === E.sel) {
                c.fillStyle = '#fff';
                for (const [hx, hy] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]) c.fillRect(hx - 4, hy - 4, 8, 8);
            }
        });
    }

    function hitHandle(px, py) {
        if (E.sel < 0) return null;
        const b = E.boxes[E.sel];
        const corners = { tl: toPx(b.x1, b.y1), tr: toPx(b.x2, b.y1), bl: toPx(b.x1, b.y2), br: toPx(b.x2, b.y2) };
        for (const [k, [hx, hy]] of Object.entries(corners)) if (Math.abs(px - hx) <= 7 && Math.abs(py - hy) <= 7) return k;
        return null;
    }
    function hitBox(px, py) {  // smallest box under the cursor (pigeons overlap)
        const [x, y] = toImg(px, py); let best = -1, area = Infinity;
        E.boxes.forEach((b, i) => {
            const a = (b.x2 - b.x1) * (b.y2 - b.y1);
            if (x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2 && a < area) { best = i; area = a; }
        });
        return best;
    }
    const pos = e => { const r = E.cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };

    function down(e) {
        const [px, py] = pos(e);
        if (e.button === 2 || e.button === 1) { E.drag = { mode: 'pan', px, py, panX: E.panX, panY: E.panY }; return; }
        const h = hitHandle(px, py);
        if (h) { E.drag = { mode: 'resize', h }; return; }
        const i = hitBox(px, py);
        if (i >= 0) {
            E.sel = i; E.el.querySelector('#cc-class').value = String(E.boxes[i].c);
            const [x, y] = toImg(px, py); const b = E.boxes[i];
            E.drag = { mode: 'move', x, y, b0: { ...b } };
        } else {
            const [x, y] = toImg(px, py);
            E.boxes.push({ c: E.cls, x1: x, y1: y, x2: x, y2: y }); E.sel = E.boxes.length - 1;
            E.el.querySelector('#cc-class').value = String(E.cls);
            E.drag = { mode: 'new' }; E.dirty = true;
        }
        draw();
    }
    function move(e) {
        if (!E.open || !E.drag) return;
        const [px, py] = pos(e);
        if (E.drag.mode === 'pan') { E.panX = E.drag.panX + px - E.drag.px; E.panY = E.drag.panY + py - E.drag.py; draw(); return; }
        const [x, y] = toImg(px, py).map(v => Math.max(0, Math.min(1, v))); const b = E.boxes[E.sel];
        if (E.drag.mode === 'move') {
            const dx = x - E.drag.x, dy = y - E.drag.y, b0 = E.drag.b0;
            const w = b0.x2 - b0.x1, h = b0.y2 - b0.y1;
            b.x1 = Math.max(0, Math.min(1 - w, b0.x1 + dx)); b.y1 = Math.max(0, Math.min(1 - h, b0.y1 + dy));
            b.x2 = b.x1 + w; b.y2 = b.y1 + h;
        } else if (E.drag.mode === 'new') { b.x2 = x; b.y2 = y; }
        else if (E.drag.mode === 'resize') {
            if (E.drag.h.includes('l')) b.x1 = x; else b.x2 = x;
            if (E.drag.h.includes('t')) b.y1 = y; else b.y2 = y;
        }
        E.dirty = true; draw();
    }
    function up() {
        if (!E.open || !E.drag) return;
        if (E.sel >= 0 && E.drag.mode !== 'pan') {
            const b = E.boxes[E.sel];
            [b.x1, b.x2] = [Math.min(b.x1, b.x2), Math.max(b.x1, b.x2)];
            [b.y1, b.y2] = [Math.min(b.y1, b.y2), Math.max(b.y1, b.y2)];
            if ((b.x2 - b.x1) * E.img.width * E.zoom < 4 || (b.y2 - b.y1) * E.img.height * E.zoom < 4) {  // a click, not a box
                if (E.drag.mode === 'new') { E.boxes.splice(E.sel, 1); E.sel = -1; }
            }
        }
        E.drag = null; draw();
    }
    function wheel(e) {
        e.preventDefault();
        const [px, py] = pos(e), [x, y] = toImg(px, py);
        E.zoom *= e.deltaY < 0 ? 1.2 : 1 / 1.2;
        E.panX = px - x * E.img.width * E.zoom; E.panY = py - y * E.img.height * E.zoom;
        draw();
    }
    function delSel() { if (E.sel >= 0) { E.boxes.splice(E.sel, 1); E.sel = -1; E.dirty = true; draw(); } }
    function msg(t) { if (E.el) E.el.querySelector('#cc-msg').textContent = t; }

    // capture phase: while the editor is open, keys go only to the editor (not to app.js shortcuts)
    window.addEventListener('keydown', e => {
        if (E.open) {
            if (e.target.matches && e.target.matches('select')) { if (e.key === 'Escape') e.target.blur(); else return; }
            const k = e.key;
            if (k === 'Escape') close();
            else if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 's') save();
            else if (k === 'Delete' || k === 'Backspace') delSel();
            else if (k === 'Tab') { if (E.boxes.length) { E.sel = (E.sel + 1) % E.boxes.length; E.el.querySelector('#cc-class').value = String(E.boxes[E.sel].c); zoomTo(E.boxes[E.sel]); draw(); } }
            else if (k === '0') { fit(); draw(); }
            else if (/^[1-9]$/.test(k) && +k <= names().length) {
                E.cls = +k - 1; E.el.querySelector('#cc-class').value = String(E.cls);
                if (E.sel >= 0) { E.boxes[E.sel].c = E.cls; E.dirty = true; draw(); }
            } else return;
            e.preventDefault(); e.stopPropagation(); return;
        }
        const k = e.key.toLowerCase();
        // Suspected Mislabels: A / D move through the list whatever has focus (a clicked dropdown or button keeps
        // focus and app.js then ignores letter keys); R / K / X / F take focus back from a dropdown first.
        if (activeTab() === 'mislabels' && !(e.target.matches && e.target.matches('input, textarea')) && !e.ctrlKey && !e.altKey && !e.metaKey) {
            if (k === 'a' || k === 'd') {
                const cards = [...document.querySelectorAll('#mis-list-container .item-list-card')];
                if (cards.length) {
                    let i = cards.findIndex(c => c.classList.contains('active'));
                    i = i < 0 ? 0 : Math.max(0, Math.min(cards.length - 1, i + (k === 'd' ? 1 : -1)));
                    if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
                    const fm = document.getElementById('frame-modal'); if (fm) fm.classList.add('hidden');
                    cards[i].click(); cards[i].scrollIntoView({ block: 'nearest' });
                }
                e.preventDefault(); e.stopPropagation(); return;
            }
            if (['r', 'k', 'x', 'f', 'delete'].includes(k) && e.target.matches && e.target.matches('select, button')) {
                e.preventDefault(); e.stopPropagation(); e.target.blur();
                document.dispatchEvent(new KeyboardEvent('keydown', { key: e.key, bubbles: true }));  // let app.js handle it
                return;
            }
        }
        // E = edit boxes: of the image in the full-image view, else of the selected box in Suspected Mislabels
        if (k !== 'e' || (e.target.matches && e.target.matches('input, select, textarea'))) return;
        const fm = document.getElementById('frame-modal');
        let id = null;
        if (fm && !fm.classList.contains('hidden')) { id = frameId(); fm.classList.add('hidden'); }
        else if (activeTab() === 'mislabels') id = (document.getElementById('mis-img-main') || {}).dataset?.frame;
        if (id) { open(id); e.preventDefault(); e.stopPropagation(); }
    }, true);

    // ======================================================================== keep your place across reloads
    // Saving in the box editor and "Save Changes to Dataset" reload the page; come back to the same tab and, in
    // Suspected Mislabels, to the same position in the list (= the next box, since the fixed one has left the list).
    const activeTab = () => (document.querySelector('.menu-item.active[data-tab]') || {}).dataset?.tab;
    window.addEventListener('beforeunload', () => {
        const cards = [...document.querySelectorAll('#mis-list-container .item-list-card')];
        const idx = cards.findIndex(c => c.classList.contains('active'));
        sessionStorage.setItem('cc-place', JSON.stringify({ tab: activeTab(), misIdx: idx, t: Date.now() }));
    });
    function restorePlace() {
        let p; try { p = JSON.parse(sessionStorage.getItem('cc-place') || 'null'); } catch (e) { p = null; }
        sessionStorage.removeItem('cc-place');
        if (!p || !p.tab || p.tab === 'overview' || Date.now() - p.t > 60000) return;
        const t0 = Date.now();
        (function wait() {  // app.js loads the report asynchronously; wait until its menu and lists are ready
            const btn = document.querySelector(`.menu-item[data-tab="${p.tab}"]`);
            const ready = document.querySelector('#class-distribution-list .bar-row');  // app.js rendered the report
            if (!btn || !ready) {
                if (Date.now() - t0 < 8000) setTimeout(wait, 150); return;
            }
            btn.click();
            if (p.tab !== 'mislabels' || p.misIdx < 0) return;
            (function pick() {
                const cards = document.querySelectorAll('#mis-list-container .item-list-card');
                if (!cards.length) { if (Date.now() - t0 < 8000) setTimeout(pick, 150); return; }
                cards[Math.min(p.misIdx, cards.length - 1)].click();
            })();
        })();
    }

    function frameId() {
        const src = (document.getElementById('frame-img') || {}).src || '';
        return new URL(src, location.href).searchParams.get('id');
    }

    // ======================================================================== rare-class gallery
    let R = null;
    function rareClasses() {
        const cnt = {};
        (report().items || []).forEach(it => { if (!it.removed) cnt[it.class_id] = (cnt[it.class_id] || 0) + 1; });
        const total = Object.values(cnt).reduce((a, b) => a + b, 0);
        // every class with under 10 % of the boxes, rarest first
        return Object.entries(cnt).filter(([, n]) => n < 0.10 * total).sort((a, b) => a[1] - b[1]).map(([c, n]) => [+c, n]);
    }
    function openRare(cls) {
        if (!R) {
            R = document.createElement('div'); R.className = 'cc-overlay';
            R.innerHTML = `<div class="cc-bar"><b>Rare classes</b><span class="cc-hint">Every box of the less common classes.
                Click a box to edit all boxes of its image.</span><button id="cc-rare-close">Close (Esc)</button></div>
                <div class="cc-tabs" id="cc-rare-tabs"></div><div class="cc-grid" id="cc-rare-grid"></div>`;
            document.body.appendChild(R);
            R.querySelector('#cc-rare-close').onclick = () => { R.style.display = 'none'; };
            window.addEventListener('keydown', e => { if (!E.open && R.style.display !== 'none' && e.key === 'Escape') R.style.display = 'none'; });
        }
        const rc = rareClasses();
        if (!rc.length) { alert('No rare classes: every class has at least 10 % of the boxes.'); return; }
        cls = cls ?? rc[0][0];
        R.querySelector('#cc-rare-tabs').innerHTML = rc.map(([c, n]) =>
            `<button class="${c === cls ? 'active' : ''}" data-c="${c}" style="background:#161b22;color:${color(c)};border:1px solid ${color(c)};border-radius:6px;padding:6px 12px;cursor:pointer">${esc(names()[c])} (${n})</button>`).join('');
        R.querySelectorAll('#cc-rare-tabs button').forEach(b => { b.onclick = () => openRare(+b.dataset.c); });
        const items = (report().items || []).filter(it => !it.removed && it.class_id === cls);
        R.querySelector('#cc-rare-grid').innerHTML = items.map(it =>
            `<div class="cc-card" data-id="${esc(it.id)}"><img src="${esc(it.path)}" loading="lazy">${esc(it.split)} · ${esc(it.image)}</div>`).join('');
        R.querySelectorAll('.cc-card').forEach(card => { card.onclick = () => open(card.dataset.id); });
        R.style.display = 'flex';
    }

    window.addEventListener('DOMContentLoaded', () => {  // runs after app.js has wired its own menu buttons
        if (!report()) return;
        const nav = document.querySelector('.sidebar-menu');
        if (nav) {
            const b = document.createElement('button');
            b.className = 'menu-item';
            b.innerHTML = `<i class="fa-solid fa-paw"></i> <span>5. Rare classes</span><span class="badge badge-mis">${rareClasses().reduce((a, [, n]) => a + n, 0)}</span>`;
            b.onclick = () => openRare();
            nav.appendChild(b);
        }
        const fm = document.getElementById('frame-modal');
        if (fm) {
            const btn = document.createElement('button');
            btn.className = 'cc-editbtn'; btn.textContent = '✎ Edit boxes (E)';
            btn.addEventListener('click', e => { e.stopPropagation(); const id = frameId(); fm.classList.add('hidden'); if (id) open(id); });
            fm.appendChild(btn);
        }
        const done = sessionStorage.getItem('cc-after-edit');
        if (done) { sessionStorage.removeItem('cc-after-edit'); toast(done); }
        restorePlace();
    });

    function toast(t) {  // short, non-blocking message (an alert would block the keyboard flow)
        const d = document.createElement('div');
        d.textContent = t;
        d.style.cssText = 'position:fixed;bottom:18px;left:50%;transform:translateX(-50%);z-index:3000;background:#10b981;color:#04130d;padding:10px 16px;border-radius:8px;font-size:13px;max-width:80vw';
        document.body.appendChild(d); setTimeout(() => d.remove(), 4000);
    }

    window.CleanCropEditor = { open, openRare };
})();
