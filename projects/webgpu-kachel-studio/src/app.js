/* ==========================================================================
   WEBGPU KACHEL STUDIO & 3D ROOM APPLICATION CONTROLLER
   ========================================================================== */

import { ShaderEngine, PRESETS } from './webgpu/shaderEngine.js';
import { Room3DManager } from './room/room3d.js';
import { PromptPopover } from './components/PromptPopover.js';
import { HistoryPanel } from './components/HistoryPanel.js';
import { GameMode } from './components/GameMode.js';

class App {
  constructor() {
    this.shaderEngine = new ShaderEngine();
    this.room3D = null;
    this.popover = null;
    this.historyPanel = null;
    this.gameMode = null;

    this.activePreset = PRESETS[0];
    this.activeConfig = {
      presetId: PRESETS[0].id,
      presetName: PRESETS[0].name,
      promptKeywords: PRESETS[0].promptKeywords,
      scale: PRESETS[0].scale,
      complexity: PRESETS[0].complexity,
      grout: PRESETS[0].grout,
      roughness: PRESETS[0].roughness,
      bump: PRESETS[0].bump,
      colors: [...PRESETS[0].colors],
      seed: 12345
    };

    // Multi-tile cell matrix for individual tile variations!
    this.targetMode = 'single'; // 'single' | 'multi' | 'global'
    this.gridMatrix = [
      [ { ...this.activeConfig, seed: 101 }, { ...this.activeConfig, seed: 102 }, { ...this.activeConfig, seed: 103 } ],
      [ { ...this.activeConfig, seed: 104 }, { ...this.activeConfig, seed: 105 }, { ...this.activeConfig, seed: 106 } ],
      [ { ...this.activeConfig, seed: 107 }, { ...this.activeConfig, seed: 108 }, { ...this.activeConfig, seed: 109 } ]
    ];
    this.activeCellIndex = { r: 1, c: 1 };

    this.isTileGridMode = false;
    this.studioCanvas = document.getElementById('webgpu-canvas');
    this.miniCanvas = document.getElementById('mini-kachel-canvas');
  }

  async init() {
    // 1. Initialize Shader Engine (WebGPU or Procedural Math Fallback)
    const isGPU = await this.shaderEngine.init();
    this.updateGPUStatusBadge(isGPU);

    // 2. Setup Preset Buttons
    this.renderPresetButtons();

    // 3. Initialize History Panel
    this.historyPanel = new HistoryPanel({
      shaderEngine: this.shaderEngine,
      onSelectTileCallback: (config) => this.loadConfigAndApplyToRoom(config)
    });

    // 4. Initialize 3D Room Renderer (with click & drag-drop callbacks)
    const roomContainer = document.getElementById('room-3d-container');
    this.room3D = new Room3DManager(
      roomContainer, 
      (surfaceName, point, uv) => {
        // On surface click in 3D scene -> open popover prompt
        this.popover.show(this.activeConfig, surfaceName);
      },
      (surfaceName, tileConfig, uv) => {
        // On Drag & Drop tile onto 3D room surface
        this.loadConfigAndApplyToRoom(tileConfig);
        alert(`🎯 Dropped tile directly onto 3D ${surfaceName}!`);
      }
    );
    this.room3D.init();

    // 5. Initialize Prompt Popover
    this.popover = new PromptPopover({
      shaderEngine: this.shaderEngine,
      onMutateCallback: (mutatedConfig) => {
        this.activeConfig = { ...mutatedConfig };
        this.renderActiveTile();
        this.updateRoomTextures();
      },
      onApplyRoomCallback: (config) => {
        this.activeConfig = { ...config };
        this.switchTab('room');
        this.renderActiveTile();
        this.updateRoomTextures();
      },
      onSaveHistoryCallback: (config) => {
        this.saveCurrentTileToHistory();
      }
    });

    // 6. Initialize Game Mode
    this.gameMode = new GameMode({
      shaderEngine: this.shaderEngine,
      onApplyTileToRoomCallback: (config) => this.loadConfigAndApplyToRoom(config)
    });

    // 7. Bind UI Events
    this.bindUIEvents();

    // 8. Render Initial Tile & Apply to 3D Room
    this.renderActiveTile();
    this.updateRoomTextures();
  }

  updateGPUStatusBadge(isGPU) {
    const badge = document.getElementById('gpu-badge');
    const text = document.getElementById('gpu-text');

    if (isGPU) {
      badge.className = 'gpu-indicator webgpu';
      text.textContent = 'WebGPU Active';
    } else {
      badge.className = 'gpu-indicator fallback';
      text.textContent = 'WebGPU / Canvas Pipeline';
    }
  }

  renderPresetButtons() {
    const container = document.getElementById('preset-buttons');
    container.innerHTML = '';

    PRESETS.forEach(preset => {
      const btn = document.createElement('button');
      btn.className = `preset-btn ${preset.id === this.activePreset.id ? 'active' : ''}`;
      btn.innerHTML = `
        <span>${preset.name}</span>
        <span style="font-size:0.68rem; color: var(--text-muted);">${preset.category}</span>
      `;

      btn.addEventListener('click', () => {
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        this.activePreset = preset;
        this.activeConfig = {
          presetId: preset.id,
          presetName: preset.name,
          promptKeywords: preset.promptKeywords,
          scale: preset.scale,
          complexity: preset.complexity,
          grout: preset.grout,
          roughness: preset.roughness,
          bump: preset.bump,
          colors: [...preset.colors],
          seed: Math.floor(Math.random() * 999999)
        };

        this.syncControlsWithConfig();
        this.renderActiveTile();
        this.updateRoomTextures();
      });

      container.appendChild(btn);
    });
  }

  syncControlsWithConfig() {
    document.getElementById('active-preset-name').textContent = this.activeConfig.presetName;
    document.getElementById('prompt-input').value = this.activeConfig.promptKeywords;
    
    document.getElementById('slider-scale').value = this.activeConfig.scale;
    document.getElementById('val-scale').textContent = this.activeConfig.scale.toFixed(1);

    document.getElementById('slider-complexity').value = this.activeConfig.complexity;
    document.getElementById('val-complexity').textContent = this.activeConfig.complexity.toFixed(1);

    document.getElementById('slider-grout').value = this.activeConfig.grout;
    document.getElementById('val-grout').textContent = this.activeConfig.grout.toFixed(3);

    document.getElementById('slider-roughness').value = this.activeConfig.roughness;
    document.getElementById('val-roughness').textContent = this.activeConfig.roughness.toFixed(2);

    document.getElementById('slider-bump').value = this.activeConfig.bump;
    document.getElementById('val-bump').textContent = this.activeConfig.bump.toFixed(2);

    document.getElementById('color-primary').value = this.activeConfig.colors[0];
    document.getElementById('color-secondary').value = this.activeConfig.colors[1];
    document.getElementById('color-accent').value = this.activeConfig.colors[2];
    document.getElementById('color-grout').value = this.activeConfig.colors[3];

    document.getElementById('mini-tile-name').textContent = this.activeConfig.presetName;
  }

  bindUIEvents() {
    // Navigation Tabs
    const navBtns = document.querySelectorAll('.nav-btn');
    navBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        this.switchTab(btn.dataset.tab);
      });
    });

    // Target Mode Buttons (Single Tile vs Multi / Grid vs All Tiles)
    const targetModeBtns = document.querySelectorAll('#target-mode-toggle .segmented-btn');
    targetModeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        targetModeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.targetMode = btn.dataset.mode;
        if (this.targetMode === 'multi' && !this.isTileGridMode) {
          this.isTileGridMode = true;
          document.getElementById('tile-toggle-label').textContent = 'Single Tile View';
        }
        this.renderActiveTile();
      });
    });

    // Randomize Seed
    document.getElementById('quick-randomize-btn').addEventListener('click', () => {
      if (this.targetMode === 'single') {
        const { r, c } = this.activeCellIndex;
        this.gridMatrix[r][c].seed = Math.floor(Math.random() * 999999);
        this.activeConfig.seed = this.gridMatrix[r][c].seed;
      } else {
        this.activeConfig.seed = Math.floor(Math.random() * 999999);
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) {
            this.gridMatrix[r][c].seed = Math.floor(Math.random() * 999999);
          }
        }
      }
      this.renderActiveTile();
      this.updateRoomTextures();
    });

    // HTML5 Drag & Drop onto 2D Studio Canvas
    this.studioCanvas.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.studioCanvas.style.outline = '2px dashed #00f2fe';
    });

    this.studioCanvas.addEventListener('dragleave', () => {
      this.studioCanvas.style.outline = 'none';
    });

    this.studioCanvas.addEventListener('drop', (e) => {
      e.preventDefault();
      this.studioCanvas.style.outline = 'none';
      const raw = e.dataTransfer.getData('application/json');
      if (raw) {
        try {
          const droppedConfig = JSON.parse(raw);

          if (this.isTileGridMode) {
            const rect = this.studioCanvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;
            const c = Math.min(2, Math.floor((clickX / rect.width) * 3));
            const r = Math.min(2, Math.floor((clickY / rect.height) * 3));

            this.gridMatrix[r][c] = { ...droppedConfig };
            this.activeCellIndex = { r, c };
            this.activeConfig = { ...droppedConfig };
          } else {
            this.activeConfig = { ...droppedConfig };
            for (let r = 0; r < 3; r++) {
              for (let c = 0; c < 3; c++) {
                this.gridMatrix[r][c] = { ...droppedConfig, seed: (r * 3 + c) * 111 };
              }
            }
          }

          this.syncControlsWithConfig();
          this.renderActiveTile();
          this.updateRoomTextures();
        } catch (err) {
          console.warn('Drop error', err);
        }
      }
    });

    // Click / Touch directly on tile canvas -> Identify clicked cell or open popover
    const handleCanvasInteraction = (e) => {
      const rect = this.studioCanvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;

      const clickX = clientX - rect.left;
      const clickY = clientY - rect.top;

      if (this.isTileGridMode) {
        const c = Math.min(2, Math.floor((clickX / rect.width) * 3));
        const r = Math.min(2, Math.floor((clickY / rect.height) * 3));
        this.activeCellIndex = { r, c };
        this.activeConfig = { ...this.gridMatrix[r][c] };

        if (this.targetMode === 'single') {
          // Mutate only this single cell on click!
          this.gridMatrix[r][c].seed = Math.floor(Math.random() * 999999);
          this.activeConfig.seed = this.gridMatrix[r][c].seed;
        } else {
          this.popover.show(this.activeConfig, `Grid Cell (${r + 1}, ${c + 1})`);
        }
      } else {
        this.popover.show(this.activeConfig, 'Studio Kachel');
      }

      this.renderActiveTile();
      this.updateRoomTextures();
    };

    this.studioCanvas.addEventListener('click', handleCanvasInteraction);
    this.studioCanvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      handleCanvasInteraction(e);
    }, { passive: false });

    // Prompt input submit
    document.getElementById('apply-prompt-btn').addEventListener('click', () => {
      this.activeConfig.promptKeywords = document.getElementById('prompt-input').value;
      this.activeConfig.seed = Math.floor(Math.random() * 999999);
      this.renderActiveTile();
      this.updateRoomTextures();
    });

    // Sliders
    const bindSlider = (id, key, valId, fixed = 1) => {
      const el = document.getElementById(id);
      const valEl = document.getElementById(valId);
      el.addEventListener('input', (e) => {
        const num = parseFloat(e.target.value);
        this.activeConfig[key] = num;
        if (valEl) valEl.textContent = num.toFixed(fixed);
        this.renderActiveTile();
        this.updateRoomTextures();
      });
    };

    bindSlider('slider-scale', 'scale', 'val-scale', 1);
    bindSlider('slider-complexity', 'complexity', 'val-complexity', 1);
    bindSlider('slider-grout', 'grout', 'val-grout', 3);
    bindSlider('slider-roughness', 'roughness', 'val-roughness', 2);
    bindSlider('slider-bump', 'bump', 'val-bump', 2);

    // Colors
    const bindColor = (id, idx) => {
      document.getElementById(id).addEventListener('input', (e) => {
        this.activeConfig.colors[idx] = e.target.value;
        this.renderActiveTile();
        this.updateRoomTextures();
      });
    };
    bindColor('color-primary', 0);
    bindColor('color-secondary', 1);
    bindColor('color-accent', 2);
    bindColor('color-grout', 3);

    // Save to History button
    document.getElementById('btn-save-history').addEventListener('click', () => {
      this.saveCurrentTileToHistory();
    });

    // Click directly on tile canvas -> open popover prompt
    this.studioCanvas.addEventListener('click', () => {
      this.popover.show(this.activeConfig, 'Studio Kachel');
    });

    // Toggle 3x3 Tile Grid Preview
    document.getElementById('btn-toggle-tiling').addEventListener('click', () => {
      this.isTileGridMode = !this.isTileGridMode;
      document.getElementById('tile-toggle-label').textContent = this.isTileGridMode ? 'Single Tile View' : 'Grid Preview (3x3)';
      this.renderActiveTile();
    });

    // Direct "Apply to 3D Room →"
    document.getElementById('btn-apply-to-room-direct').addEventListener('click', () => {
      this.switchTab('room');
      this.updateRoomTextures();
    });

    // 3D Room Surface Target segmented buttons
    const surfaceBtns = document.querySelectorAll('#room-surface-target .segmented-btn');
    surfaceBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        surfaceBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.room3D.activeSurfaceTarget = btn.dataset.surface;
        this.updateRoomTextures();
      });
    });

    // 3D Lighting preset segmented buttons
    const lightBtns = document.querySelectorAll('#lighting-preset .segmented-btn');
    lightBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        lightBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.room3D.setupLighting(btn.dataset.light);
      });
    });

    // 3D Room Sliders
    document.getElementById('room-tile-repeat').addEventListener('input', (e) => {
      const val = parseInt(e.target.value);
      this.room3D.tileRepeatScale = val;
      document.getElementById('val-room-repeat').textContent = `${val}x${val}`;
      this.updateRoomTextures();
    });

    // Reset 3D Camera view
    document.getElementById('btn-reset-camera').addEventListener('click', () => {
      this.room3D.resetView();
    });

    // Take photo snapshot of 3D room
    document.getElementById('btn-snapshot-room').addEventListener('click', () => {
      if (this.room3D && this.room3D.renderer) {
        const dataUrl = this.room3D.renderer.domElement.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = `room-kachel-snapshot-${Date.now()}.png`;
        link.href = dataUrl;
        link.click();
      }
    });

    // Mini re-gen button in room tab
    document.getElementById('mini-re-gen-btn').addEventListener('click', () => {
      this.activeConfig.seed = Math.floor(Math.random() * 999999);
      this.renderActiveTile();
      this.updateRoomTextures();
    });
  }

  switchTab(tabId) {
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === tabId);
    });

    document.querySelectorAll('.tab-content').forEach(content => {
      content.classList.toggle('active', content.id === `tab-${tabId}`);
    });

    if (tabId === 'room' && this.room3D) {
      setTimeout(() => this.room3D.onWindowResize(), 50);
    }
  }

  renderActiveTile() {
    if (this.isTileGridMode) {
      // Render 3x3 multi-tile grid where each cell can be different!
      this.shaderEngine.renderMultiTileGridToCanvas(this.studioCanvas, this.gridMatrix, 3, 3);
    } else {
      this.shaderEngine.renderTileToCanvas(this.studioCanvas, this.activeConfig);
    }

    // Also render mini preview canvas for sidebar
    if (this.miniCanvas) {
      this.shaderEngine.renderTileToCanvas(this.miniCanvas, this.activeConfig);
    }
  }

  updateRoomTextures() {
    if (!this.room3D) return;

    // Generate single tile canvas & normal map
    const tileCanvas = document.createElement('canvas');
    tileCanvas.width = 512;
    tileCanvas.height = 512;
    this.shaderEngine.renderTileToCanvas(tileCanvas, this.activeConfig);

    const normalCanvas = this.shaderEngine.generateNormalMapCanvas(tileCanvas);

    this.room3D.applyTileTextures(tileCanvas, normalCanvas, {
      repeatScale: this.room3D.tileRepeatScale,
      roughness: this.activeConfig.roughness
    });
  }

  saveCurrentTileToHistory() {
    if (this.historyPanel) {
      const tileCanvas = document.createElement('canvas');
      tileCanvas.width = 512;
      tileCanvas.height = 512;
      this.shaderEngine.renderTileToCanvas(tileCanvas, this.activeConfig);

      this.historyPanel.addTile(this.activeConfig, tileCanvas);
      alert(`✨ "${this.activeConfig.presetName}" saved to history gallery!`);
    }
  }

  loadConfigAndApplyToRoom(config) {
    this.activeConfig = { ...config };
    this.syncControlsWithConfig();
    this.renderActiveTile();
    this.switchTab('room');
    this.updateRoomTextures();
  }
}

// Instantiate and launch app on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new App();
  app.init();
});
