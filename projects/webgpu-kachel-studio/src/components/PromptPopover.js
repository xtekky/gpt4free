/* ==========================================================================
   ON-CLICK TILE & ROOM SURFACE INSPECTOR POPOVER
   ========================================================================== */

export class PromptPopover {
  constructor({ shaderEngine, onMutateCallback, onApplyRoomCallback, onSaveHistoryCallback }) {
    this.shaderEngine = shaderEngine;
    this.onMutate = onMutateCallback;
    this.onApplyRoom = onApplyRoomCallback;
    this.onSaveHistory = onSaveHistoryCallback;

    this.modalEl = document.getElementById('prompt-popover');
    this.canvasEl = document.getElementById('popover-canvas');
    this.promptInputEl = document.getElementById('popover-prompt-text');

    this.activeConfig = null;
    this.initEvents();
  }

  initEvents() {
    // Close button & backdrop
    const closeBtn = document.getElementById('popover-close-btn');
    const backdrop = this.modalEl.querySelector('.popover-backdrop');

    closeBtn.addEventListener('click', () => this.hide());
    backdrop.addEventListener('click', () => this.hide());

    // Randomize seed button
    document.getElementById('btn-popover-seed').addEventListener('click', () => {
      if (this.activeConfig) {
        this.activeConfig.seed = Math.floor(Math.random() * 999999);
        this.renderPreview();
        this.notifyMutate();
      }
    });

    // Quick palette buttons
    const paletteBtns = document.getElementById('popover-palette-btns').querySelectorAll('.palette-chip');
    paletteBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const hexes = btn.dataset.colors.split(',');
        if (this.activeConfig && hexes.length >= 4) {
          this.activeConfig.colors = hexes;
          this.renderPreview();
          this.notifyMutate();
        }
      });
    });

    // Sliders
    document.getElementById('popover-slider-scale').addEventListener('input', (e) => {
      if (this.activeConfig) {
        this.activeConfig.scale = parseFloat(e.target.value);
        this.renderPreview();
        this.notifyMutate();
      }
    });

    document.getElementById('popover-slider-roughness').addEventListener('input', (e) => {
      if (this.activeConfig) {
        this.activeConfig.roughness = parseFloat(e.target.value);
        this.renderPreview();
        this.notifyMutate();
      }
    });

    // Prompt input change
    this.promptInputEl.addEventListener('change', () => {
      if (this.activeConfig) {
        this.activeConfig.promptKeywords = this.promptInputEl.value;
        this.notifyMutate();
      }
    });

    // Apply to 3D room button
    document.getElementById('btn-popover-apply-room').addEventListener('click', () => {
      if (this.onApplyRoom && this.activeConfig) {
        this.onApplyRoom(this.activeConfig);
      }
      this.hide();
    });

    // Save to history
    document.getElementById('btn-popover-save-history').addEventListener('click', () => {
      if (this.onSaveHistory && this.activeConfig) {
        this.onSaveHistory(this.activeConfig);
      }
    });
  }

  show(config, surfaceName = 'Tile Surface') {
    this.activeConfig = { ...config };
    this.promptInputEl.value = this.activeConfig.promptKeywords || '';
    document.getElementById('popover-slider-scale').value = this.activeConfig.scale || 8.0;
    document.getElementById('popover-slider-roughness').value = this.activeConfig.roughness || 0.3;

    this.renderPreview();
    this.modalEl.classList.remove('hidden');
  }

  hide() {
    this.modalEl.classList.add('hidden');
  }

  renderPreview() {
    if (this.canvasEl && this.activeConfig && this.shaderEngine) {
      this.shaderEngine.renderTileToCanvas(this.canvasEl, this.activeConfig);
    }
  }

  notifyMutate() {
    if (this.onMutate && this.activeConfig) {
      this.onMutate(this.activeConfig);
    }
  }
}
