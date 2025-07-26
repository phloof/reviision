// ChartManager - Unified chart wrapper for ReViision analytics
// Dependencies: Chart.js v3+, chartjs-plugin-zoom, chartjs-plugin-datalabels, FileSaver.js
// This module registers the plugins once and exposes convenient helper methods for
// switching chart type, toggling datasets, zoom controls, and exporting the chart
// data/visualisation in PNG, CSV, or JSON formats.

(function(global) {
    "use strict";

    if (!global.Chart) {
        console.error("Chart.js not found – ChartManager requires Chart.js v3+");
        return;
    }

    // Attempt to register plugins if they exist in the global scope
    const maybeRegister = (pluginGlobalName) => {
        if (global[pluginGlobalName]) {
            try {
                global.Chart.register(global[pluginGlobalName]);
            } catch (err) {
                // Ignore if already registered
            }
        }
    };

    maybeRegister("chartjsZoom");            // chartjs-plugin-zoom attaches itself under this name
    maybeRegister("ChartDataLabels");        // chartjs-plugin-datalabels
    maybeRegister("ChartZoom"); // alternative global var name for zoom plugin

    const saveAs = global.saveAs; // Provided by FileSaver.js

    class ChartManager {
        constructor(canvasCtx, config, options = {}) {
            // Create underlying Chart.js instance
            this.canvas = canvasCtx;
            // Deep-clone scales for restoration when switching types
            this._originalScales = config.options && config.options.scales ? JSON.parse(JSON.stringify(config.options.scales)) : undefined;

            // Ensure zoom options are present by default - DISABLE wheel zoom to prevent unwanted scroll behavior
            const zoomDefaults = {
                zoom: {
                    wheel: { enabled: false }, // Disabled to prevent scroll-to-zoom issues
                    pinch: { enabled: true },
                    mode: 'xy'
                },
                pan: { enabled: false, mode: 'xy' } // Disabled by default, controlled by toggle
            };
            config.options = config.options || {};
            config.options.plugins = config.options.plugins || {};
            config.options.plugins.zoom = Object.assign({}, zoomDefaults, config.options.plugins.zoom || {});

            this.chart = new global.Chart(canvasCtx, config);
            // Merge user-provided options (e.g., default filename stub, id)
            this.options = Object.assign({ filename: "chart", toolbar: true }, options);

            // Track zoom state for limits
            this.zoomLevel = 1.0; // Track current zoom level
            this.minZoom = 1.0;   // Minimum zoom (original size)
            this.maxZoom = 5.0;   // Maximum zoom level
            this.panEnabled = false; // Track pan mode state

            if (this.options.toolbar) {
                this._buildToolbar();
            }
        }

        // ---------- Core helpers ---------- //
        switchType(newType) {
            if (this.chart.config.type === newType) return;
            this.chart.config.type = newType;

            // Axes handling: remove for radial charts, restore for cartesian
            const radialTypes = ['doughnut', 'pie', 'polarArea', 'radar'];
            if (radialTypes.includes(newType)) {
                this.chart.options.scales = {};
            } else {
                if (this._originalScales) {
                    this.chart.options.scales = JSON.parse(JSON.stringify(this._originalScales));
                }
            }

            this.chart.update();
        }

        toggleDataset(dsIndex) {
            // Chart.js v3+ has built-in helpers
            if (this.chart.isDatasetVisible(dsIndex)) {
                this.chart.hide(dsIndex);
            } else {
                this.chart.show(dsIndex);
            }
        }

        resetZoom() {
            if (this.chart.resetZoom) {
                this.chart.resetZoom();
                this.zoomLevel = 1.0;
            }
        }

        // New zoom in method with limits
        zoomIn() {
            if (this.zoomLevel < this.maxZoom && this.chart.zoom) {
                const zoomFactor = 1.2;
                this.chart.zoom(zoomFactor);
                this.zoomLevel *= zoomFactor;
            }
        }

        // New zoom out method with limits
        zoomOut() {
            if (this.zoomLevel > this.minZoom && this.chart.zoom) {
                const zoomFactor = 0.8;
                // Don't zoom out beyond original size
                const newZoomLevel = this.zoomLevel * zoomFactor;
                if (newZoomLevel >= this.minZoom) {
                    this.chart.zoom(zoomFactor);
                    this.zoomLevel = newZoomLevel;
                } else {
                    // Reset to original size if we would go below minimum
                    this.resetZoom();
                }
            }
        }

        // Toggle pan mode
        togglePan() {
            this.panEnabled = !this.panEnabled;
            if (this.chart.options.plugins.zoom) {
                this.chart.options.plugins.zoom.pan.enabled = this.panEnabled;
                this.chart.update('none'); // Update without animation

                // Update button visual state and canvas cursor
                const toolbar = this.canvas.closest('.chart-container')?.querySelector('.chart-toolbar');
                const panBtn = toolbar?.querySelector('[data-action="toggle-pan"]');
                if (panBtn) {
                    if (this.panEnabled) {
                        panBtn.classList.add('active');
                        panBtn.title = 'Disable Pan Mode';
                        this.canvas.classList.add('pan-active');
                    } else {
                        panBtn.classList.remove('active');
                        panBtn.title = 'Enable Pan Mode';
                        this.canvas.classList.remove('pan-active');
                    }
                }
            }
        }

        toggleToolbar() {
            const container = this.canvas.closest('.chart-container') || this.canvas.parentElement;
            const toolbar = container.querySelector('.chart-toolbar');
            const expandBtn = toolbar.querySelector('.chart-expand-btn');
            const expandIcon = expandBtn.querySelector('i');

            if (toolbar.classList.contains('collapsed')) {
                // Expand toolbar
                toolbar.classList.remove('collapsed');
                expandIcon.className = 'fas fa-chevron-right';
                expandBtn.title = 'Hide Chart Controls';
            } else {
                // Collapse toolbar
                toolbar.classList.add('collapsed');
                expandIcon.className = 'fas fa-chevron-left';
                expandBtn.title = 'Show Chart Controls';
            }
        }

        resize() {
            this.chart.resize();
        }

        // ---------- Export helpers ---------- //
        exportPNG(filename) {
            const name = filename || `${this.options.filename}.png`;
            // Ensure the chart is rendered at current size
            this.chart.resize();
            this.canvas.toBlob((blob) => {
                if (blob && saveAs) {
                    saveAs(blob, name);
                } else {
                    console.error("FileSaver.js saveAs function missing or canvas blob failed");
                }
            }, "image/png", 1.0);
        }

        exportCSV(filename) {
            const name = filename || `${this.options.filename}.csv`;
            const { labels = [] } = this.chart.data;
            const datasets = this.chart.data.datasets || [];

            // Build CSV rows (header + data rows)
            const rows = [];
            const header = ["Label", ...datasets.map((d) => sanitizeCSV(d.label))];
            rows.push(header.join(","));

            labels.forEach((lbl, idx) => {
                const row = [sanitizeCSV(lbl)];
                datasets.forEach((ds) => {
                    row.push(ds.data[idx] != null ? ds.data[idx] : "");
                });
                rows.push(row.join(","));
            });

            const csvBlob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" });
            saveAs(csvBlob, name);
        }

        exportJSON(filename) {
            const name = filename || `${this.options.filename}.json`;
            const json = JSON.stringify({ labels: this.chart.data.labels, datasets: this.chart.data.datasets }, null, 2);
            const blob = new Blob([json], { type: "application/json;charset=utf-8" });
            saveAs(blob, name);
        }

        destroy() {
            this.chart.destroy();
        }

        // Build toolbar UI and attach event listeners
        _buildToolbar() {
            const container = this.canvas.closest('.chart-container') || this.canvas.parentElement;
            if (!container) return;
            // Ensure container is positioned for absolute toolbar
            if (getComputedStyle(container).position === 'static') {
                container.style.position = 'relative';
            }
            // Avoid duplicates
            if (container.querySelector('.chart-toolbar')) return;

            // Inject toolbar CSS once
            if (!document.getElementById('chart-toolbar-style')) {
                const style = document.createElement('style');
                style.id = 'chart-toolbar-style';
                style.textContent = `
                     .chart-toolbar { position: absolute; top: -2px; right: 10px; z-index: 40; display: flex; gap: 4px; flex-wrap: nowrap; background: rgba(255,255,255,0.85); border-radius: 4px; padding: 2px 4px; box-shadow: 0 0 4px rgba(0,0,0,0.1); transition: transform 0.3s ease, opacity 0.3s ease; }
                     .chart-toolbar.collapsed { transform: translateX(calc(100% - 36px)); }
                     .chart-toolbar.collapsed .chart-controls { opacity: 0; pointer-events: none; transform: translateX(20px); }
                     .chart-toolbar .chart-controls { display: flex; gap: 4px; transition: all 0.3s ease; }
                     .chart-toolbar .chart-expand-btn { order: -1; margin-right: 4px; border-radius: 4px 0 0 4px; }
                     .chart-toolbar .btn { padding: 2px 6px; font-size: 0.75rem; line-height: 1; transition: all 0.2s ease; }
                     .chart-toolbar .btn.active { background-color: #0d6efd; color: white; border-color: #0d6efd; }
                     .chart-toolbar .btn:hover { transform: translateY(-1px); }
                     .chart-toolbar .btn-group:last-child .btn { border-radius: 0 4px 4px 0; }
                     .chart-toolbar .dropdown-menu { font-size: 0.875rem; min-width: 140px; box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15); }
                     .chart-toolbar .dropdown-item { padding: 0.375rem 0.75rem; transition: background-color 0.15s ease-in-out; }
                     .chart-toolbar .dropdown-item:hover { background-color: #f8f9fa; }
                     .chart-toolbar .dropdown-item i { width: 16px; text-align: center; }
                     .chart-container { overflow: visible; } /* prevent tooltip clipping */
                     .card-header { overflow: visible; }
                     .tooltip .tooltip-text { left: auto; right: 110%; top: 50%; transform: translateY(-50%); white-space: normal; max-width: 220px; }
                     /* Pan mode cursor styles */
                     .chart-container canvas.pan-active { cursor: grab !important; }
                     .chart-container canvas.pan-active:active { cursor: grabbing !important; }
                 `;
                document.head.appendChild(style);
            }

            const TYPES = ['bar', 'line', 'doughnut', 'pie', 'radar'];

            const toolbar = document.createElement('div');
            toolbar.className = 'chart-toolbar text-end mt-1 collapsed';
            toolbar.innerHTML = `
                <button class="btn btn-sm btn-outline-secondary chart-expand-btn" data-action="toggle-toolbar" title="Show Chart Controls">
                    <i class="fas fa-chevron-left"></i>
                </button>
                <div class="chart-controls">
                    <div class="btn-group me-1">
                        <button class="btn btn-sm btn-outline-secondary dropdown-toggle" data-bs-toggle="dropdown" aria-expanded="false" title="Change Chart Type"><i class="fas fa-random"></i></button>
                        <ul class="dropdown-menu dropdown-menu-end">
                            ${TYPES.map(t => `<li><a class="dropdown-item" data-type="${t}" href="#">${t.charAt(0).toUpperCase() + t.slice(1)}</a></li>`).join('')}
                        </ul>
                    </div>
                    <button class="btn btn-sm btn-outline-secondary me-1" data-action="zoom-in" title="Zoom In"><i class="fas fa-search-plus"></i></button>
                    <button class="btn btn-sm btn-outline-secondary me-1" data-action="zoom-out" title="Zoom Out"><i class="fas fa-search-minus"></i></button>
                    <button class="btn btn-sm btn-outline-secondary me-1" data-action="reset-zoom" title="Reset Zoom"><i class="fas fa-expand-arrows-alt"></i></button>
                    <button class="btn btn-sm btn-outline-secondary me-1" data-action="toggle-pan" title="Enable Pan Mode"><i class="fas fa-hand-paper"></i></button>
                    <div class="btn-group">
                        <button class="btn btn-sm btn-outline-secondary dropdown-toggle" data-bs-toggle="dropdown" aria-expanded="false" title="Export Options">
                            <i class="fas fa-download"></i>
                        </button>
                        <ul class="dropdown-menu dropdown-menu-end">
                            <li><a class="dropdown-item" data-action="png" href="#"><i class="fas fa-image me-2"></i>PNG Image</a></li>
                            <li><a class="dropdown-item" data-action="csv" href="#"><i class="fas fa-file-csv me-2"></i>CSV Data</a></li>
                            <li><a class="dropdown-item" data-action="json" href="#"><i class="fas fa-code me-2"></i>JSON Data</a></li>
                        </ul>
                    </div>
                </div>
            `;

            container.appendChild(toolbar);

            // Event delegation for toolbar actions
            toolbar.addEventListener('click', (ev) => {
                const typeItem = ev.target.closest('[data-type]');
                if (typeItem) {
                    ev.preventDefault();
                    const newType = typeItem.getAttribute('data-type');
                    this.switchType(newType);
                    return;
                }

                const btn = ev.target.closest('[data-action]');
                if (!btn) return;

                ev.preventDefault();
                const action = btn.getAttribute('data-action');

                // Handle export actions
                if (action === 'png') {
                    this.exportPNG();
                } else if (action === 'csv') {
                    this.exportCSV();
                } else if (action === 'json') {
                    this.exportJSON();
                }
                // Handle zoom actions
                else if (action === 'zoom-in') {
                    this.zoomIn();
                } else if (action === 'zoom-out') {
                    this.zoomOut();
                } else if (action === 'reset-zoom') {
                    this.resetZoom();
                }
                // Handle pan toggle
                else if (action === 'toggle-pan') {
                    this.togglePan();
                }
                // Handle toggle toolbar
                else if (action === 'toggle-toolbar') {
                    this.toggleToolbar();
                }
            });
        }
    }

    function sanitizeCSV(value) {
        // Escape quotes / commas
        if (typeof value === "string" && (value.includes(",") || value.includes("\""))) {
            return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
    }

    // Expose globally
    global.ChartManager = ChartManager;
})(window); 