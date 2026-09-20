/* ==========================================================================
   KACHEL ARCADE & SPATIAL ROOM DECORATOR GAME ENGINE
   ========================================================================== */

import { PRESETS } from '../webgpu/shaderEngine.js';

export class GameMode {
  constructor({ shaderEngine, onApplyTileToRoomCallback }) {
    this.shaderEngine = shaderEngine;
    this.onApplyTileToRoom = onApplyTileToRoomCallback;

    this.stageContainer = document.getElementById('game-stage-container');
    this.scoreEl = document.getElementById('game-score');
    this.levelEl = document.getElementById('game-level');

    this.activeMode = 'matcher'; // 'matcher' | 'memory'
    this.score = 0;
    this.level = 1;

    this.matcherState = null;
    this.memoryState = null;

    this.initModeSelector();
  }

  initModeSelector() {
    const cards = document.querySelectorAll('.game-mode-card');
    cards.forEach(card => {
      card.addEventListener('click', () => {
        cards.forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        this.activeMode = card.dataset.mode;
        this.startSelectedMode();
      });
    });

    this.startSelectedMode();
  }

  startSelectedMode() {
    if (this.activeMode === 'matcher') {
      this.initMatcherGame();
    } else if (this.activeMode === 'dragdrop') {
      this.initDragDropPainterGame();
    } else {
      this.initMemoryGame();
    }
  }

  // --- GAME MODE 1: ROOM AESTHETIC MATCHER ---
  initMatcherGame() {
    const randomPreset = PRESETS[Math.floor(Math.random() * PRESETS.length)];
    const targetConfig = {
      ...randomPreset,
      presetId: randomPreset.id,
      scale: Math.floor(Math.random() * 12) + 4,
      complexity: Math.floor(Math.random() * 6) + 3,
      seed: Math.floor(Math.random() * 99999)
    };

    const playerConfig = {
      presetId: PRESETS[0].id,
      scale: 8,
      complexity: 5,
      grout: 0.03,
      roughness: 0.3,
      colors: [...PRESETS[0].colors],
      seed: 12345
    };

    this.matcherState = { targetConfig, playerConfig, timer: 60 };

    this.stageContainer.innerHTML = `
      <div class="matcher-game-grid" style="display: flex; gap: 2rem; width: 100%; max-width: 900px;">
        <!-- Target Room Client Request -->
        <div style="flex: 1; display: flex; flex-direction: column; gap: 1rem; background: rgba(15,20,30,0.6); padding: 1.25rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08);">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="badge badge-gold">Client Order #${Math.floor(Math.random()*900+100)}</span>
            <span style="font-family: var(--font-code); color: var(--color-gold);">⏱️ Time: <strong id="matcher-timer">60s</strong></span>
          </div>
          <h3>Client Request: "${targetConfig.name}"</h3>
          <p style="font-size: 0.82rem; color: var(--text-secondary);">"I need tiles matching these exact visual characteristics: <strong>${targetConfig.promptKeywords}</strong>."</p>
          <div style="width: 100%; aspect-ratio: 1 / 1; background: #000; border-radius: 12px; overflow: hidden; position: relative;">
            <canvas id="target-kachel-canvas" width="300" height="300" style="width: 100%; height: 100%;"></canvas>
            <span style="position: absolute; bottom: 8px; left: 8px; background: rgba(0,0,0,0.7); padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; color: #fff;">TARGET MATCH</span>
          </div>
        </div>

        <!-- Player Design Station -->
        <div style="flex: 1; display: flex; flex-direction: column; gap: 1rem; background: rgba(15,20,30,0.6); padding: 1.25rem; border-radius: 16px; border: 1px solid rgba(0,242,254,0.3);">
          <h3>Your WebGPU Design</h3>
          <div style="width: 100%; aspect-ratio: 1 / 1; background: #000; border-radius: 12px; overflow: hidden;">
            <canvas id="player-kachel-canvas" width="300" height="300" style="width: 100%; height: 100%;"></canvas>
          </div>

          <div style="display: flex; flex-direction: column; gap: 0.5rem;">
            <label style="font-size: 0.8rem; color: var(--text-secondary);">Select Pattern Style</label>
            <select id="game-preset-select" style="background: var(--bg-input); color: #fff; padding: 0.4rem; border-radius: 6px; border: 1px solid var(--bg-card-border);">
              ${PRESETS.map(p => `<option value="${p.id}">${p.name}</option>`).join('')}
            </select>

            <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
              <span>Scale:</span>
              <input type="range" id="game-slider-scale" min="1" max="20" value="${playerConfig.scale}" style="width: 60%;">
            </div>
          </div>

          <div style="display: flex; gap: 0.5rem; margin-top: auto;">
            <button id="btn-submit-match" class="btn btn-accent btn-full">Submit to Client (Check Match)</button>
          </div>
          <div id="match-result-score" style="text-align: center; font-family: var(--font-code); font-size: 1rem; color: var(--color-emerald);"></div>
        </div>
      </div>
    `;

    // Render target & initial player canvas
    const targetCanvas = document.getElementById('target-kachel-canvas');
    const playerCanvas = document.getElementById('player-kachel-canvas');

    this.shaderEngine.renderTileToCanvas(targetCanvas, targetConfig);
    this.shaderEngine.renderTileToCanvas(playerCanvas, playerConfig);

    // Event handlers
    const presetSelect = document.getElementById('game-preset-select');
    const sliderScale = document.getElementById('game-slider-scale');

    const updatePlayerTile = () => {
      playerConfig.presetId = presetSelect.value;
      const selectedP = PRESETS.find(p => p.id === presetSelect.value);
      if (selectedP) playerConfig.colors = selectedP.colors;
      playerConfig.scale = parseFloat(sliderScale.value);
      this.shaderEngine.renderTileToCanvas(playerCanvas, playerConfig);
    };

    presetSelect.addEventListener('change', updatePlayerTile);
    sliderScale.addEventListener('input', updatePlayerTile);

    document.getElementById('btn-submit-match').addEventListener('click', () => {
      const isPresetMatch = playerConfig.presetId === targetConfig.presetId;
      const scaleDiff = Math.abs(playerConfig.scale - targetConfig.scale);
      
      let accuracy = 40;
      if (isPresetMatch) accuracy += 40;
      accuracy += Math.max(0, 20 - scaleDiff * 2);

      const resultText = document.getElementById('match-result-score');
      if (accuracy >= 80) {
        resultText.innerHTML = `🎉 EXCELLENT MATCH! Accuracy: ${accuracy}% (+250 PTS)`;
        this.score += 250;
        this.level++;
        this.scoreEl.textContent = this.score;
        this.levelEl.textContent = this.level;
        setTimeout(() => this.initMatcherGame(), 2000);
      } else {
        resultText.innerHTML = `⚠️ Match Accuracy: ${accuracy}%. Try adjusting preset style or scale!`;
        resultText.style.color = '#f59e0b';
      }
    });
  }

  // --- GAME MODE 3: DRAG & DROP TILE PAINTER ARCADE ---
  initDragDropPainterGame() {
    const mosaicGrid = Array(16).fill(null);
    let activeBrushConfig = { ...PRESETS[0], seed: 101 };

    this.stageContainer.innerHTML = `
      <div style="display: flex; flex-direction: column; align-items: center; gap: 1.25rem; width: 100%; max-width: 900px;">
        <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
          <div>
            <h3>🎨 Drag & Drop Mosaic Craft</h3>
            <p style="font-size: 0.8rem; color: var(--text-secondary);">Drag tiles from your palette onto grid slots, or click slots to paint with selected tile brush!</p>
          </div>
          <button id="btn-submit-mosaic" class="btn btn-emerald btn-sm">Submit Completed Mosaic (+300 PTS)</button>
        </div>

        <div style="display: flex; gap: 2rem; width: 100%;">
          <!-- Tile Palette Side Bar -->
          <div style="width: 220px; display: flex; flex-direction: column; gap: 0.75rem; background: rgba(15,20,30,0.6); padding: 1rem; border-radius: 16px; border: 1px solid var(--bg-card-border);">
            <span class="badge badge-gold">Tile Palette (Drag Me!)</span>
            <div id="palette-tiles-list" style="display: flex; flex-direction: column; gap: 0.6rem;"></div>
          </div>

          <!-- Mosaic Canvas Grid -->
          <div style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(10,13,20,0.8); padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(0,242,254,0.3);">
            <div id="mosaic-canvas-grid" style="display: grid; grid-template-columns: repeat(4, 85px); gap: 8px; background: #06080d; padding: 12px; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.6);"></div>
            <span style="margin-top: 1rem; font-size: 0.75rem; color: var(--text-muted);">Tip: Touch or click individual grid slots to toggle tile variations!</span>
          </div>
        </div>
      </div>
    `;

    // Render Tile Palette
    const paletteContainer = document.getElementById('palette-tiles-list');
    PRESETS.slice(0, 5).forEach((preset, idx) => {
      const card = document.createElement('div');
      card.style.cssText = 'display: flex; align-items: center; gap: 0.5rem; background: #131823; padding: 0.4rem; border-radius: 8px; cursor: grab; border: 1px solid rgba(255,255,255,0.1);';
      card.setAttribute('draggable', 'true');

      const canvas = document.createElement('canvas');
      canvas.width = 45;
      canvas.height = 45;
      canvas.style.cssText = 'border-radius: 4px;';
      const config = { ...preset, seed: (idx + 1) * 333 };
      this.shaderEngine.renderTileToCanvas(canvas, config);

      const label = document.createElement('span');
      label.style.cssText = 'font-size: 0.75rem; font-weight: 600; font-family: var(--font-heading);';
      label.textContent = preset.name.split(' ')[0];

      card.appendChild(canvas);
      card.appendChild(label);

      card.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('application/json', JSON.stringify(config));
        activeBrushConfig = config;
      });

      card.addEventListener('click', () => {
        activeBrushConfig = config;
        paletteContainer.querySelectorAll('div').forEach(d => d.style.borderColor = 'rgba(255,255,255,0.1)');
        card.style.borderColor = '#00f2fe';
      });

      paletteContainer.appendChild(card);
    });

    // Render 4x4 Mosaic Grid Slots
    const mosaicGridEl = document.getElementById('mosaic-canvas-grid');
    for (let i = 0; i < 16; i++) {
      const slot = document.createElement('div');
      slot.style.cssText = 'width: 85px; height: 85px; background: rgba(255,255,255,0.03); border: 1px stroke dashed rgba(255,255,255,0.15); border-radius: 8px; cursor: pointer; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center;';

      const canvas = document.createElement('canvas');
      canvas.width = 85;
      canvas.height = 85;
      canvas.style.cssText = 'width: 100%; height: 100%; display: none;';
      slot.appendChild(canvas);

      const placeholder = document.createElement('span');
      placeholder.style.cssText = 'font-size: 0.8rem; color: #475569;';
      placeholder.textContent = `+`;
      slot.appendChild(placeholder);

      // Drag over / Drop handlers
      slot.addEventListener('dragover', (e) => {
        e.preventDefault();
        slot.style.borderColor = '#00f2fe';
      });

      slot.addEventListener('dragleave', () => {
        slot.style.borderColor = 'rgba(255,255,255,0.15)';
      });

      slot.addEventListener('drop', (e) => {
        e.preventDefault();
        slot.style.borderColor = 'rgba(255,255,255,0.15)';
        const raw = e.dataTransfer.getData('application/json');
        if (raw) {
          const cfg = JSON.parse(raw);
          mosaicGrid[i] = cfg;
          this.shaderEngine.renderTileToCanvas(canvas, cfg);
          canvas.style.display = 'block';
          placeholder.style.display = 'none';
        }
      });

      // Touch / Mouse Click Handler to paint with active brush
      const paintSlot = () => {
        if (activeBrushConfig) {
          mosaicGrid[i] = activeBrushConfig;
          this.shaderEngine.renderTileToCanvas(canvas, activeBrushConfig);
          canvas.style.display = 'block';
          placeholder.style.display = 'none';
        }
      };

      slot.addEventListener('click', paintSlot);
      slot.addEventListener('touchstart', (e) => {
        e.preventDefault();
        paintSlot();
      }, { passive: false });

      mosaicGridEl.appendChild(slot);
    }

    // Submit Mosaic Handler
    document.getElementById('btn-submit-mosaic').addEventListener('click', () => {
      const filledCount = mosaicGrid.filter(t => t !== null).length;
      if (filledCount >= 8) {
        this.score += 300;
        this.scoreEl.textContent = this.score;
        alert(`🎉 Beautiful Mosaic Completed! You placed ${filledCount} tiles (+300 PTS).`);
      } else {
        alert('Fill at least 8 grid slots with tiles before submitting!');
      }
    });
  }
  initMemoryGame() {
    // Generate 4 procedural tiles (2 copies each = 8 cards)
    const selectedPresets = PRESETS.slice(0, 4);
    const cards = [];

    selectedPresets.forEach((preset, idx) => {
      const config = { ...preset, presetId: preset.id, seed: idx * 999 };
      cards.push({ id: idx, config, flipped: false, matched: false });
      cards.push({ id: idx, config, flipped: false, matched: false });
    });

    // Shuffle cards
    cards.sort(() => Math.random() - 0.5);

    this.memoryState = { cards, flippedCards: [], matchesFound: 0 };

    this.stageContainer.innerHTML = `
      <div style="display: flex; flex-direction: column; align-items: center; gap: 1rem; width: 100%;">
        <h3>Spatial Kachel Memory - Find Matching WebGPU Tiles</h3>
        <div id="memory-grid" style="display: grid; grid-template-columns: repeat(4, 120px); gap: 1rem;"></div>
      </div>
    `;

    const gridEl = document.getElementById('memory-grid');

    cards.forEach((card, index) => {
      const cardEl = document.createElement('div');
      cardEl.className = 'memory-card';
      cardEl.style.cssText = 'width: 120px; height: 120px; background: #131823; border: 1px solid rgba(0,242,254,0.3); border-radius: 12px; cursor: pointer; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; transition: transform 0.3s;';

      const canvas = document.createElement('canvas');
      canvas.width = 120;
      canvas.height = 120;
      canvas.style.cssText = 'width: 100%; height: 100%; display: none;';
      this.shaderEngine.renderTileToCanvas(canvas, card.config);
      cardEl.appendChild(canvas);

      const cover = document.createElement('div');
      cover.style.cssText = 'font-size: 2rem; color: var(--color-primary); font-family: var(--font-heading);';
      cover.textContent = '❖';
      cardEl.appendChild(cover);

      cardEl.addEventListener('click', () => {
        if (card.flipped || card.matched || this.memoryState.flippedCards.length >= 2) return;

        card.flipped = true;
        canvas.style.display = 'block';
        cover.style.display = 'none';
        cardEl.style.borderColor = '#00f2fe';
        this.memoryState.flippedCards.push({ card, cardEl, canvas, cover });

        if (this.memoryState.flippedCards.length === 2) {
          const [c1, c2] = this.memoryState.flippedCards;
          if (c1.card.id === c2.card.id) {
            c1.card.matched = true;
            c2.card.matched = true;
            this.memoryState.matchesFound++;
            this.score += 150;
            this.scoreEl.textContent = this.score;
            this.memoryState.flippedCards = [];

            if (this.memoryState.matchesFound === 4) {
              setTimeout(() => {
                alert('🏆 Victory! You matched all spatial WebGPU kacheln!');
                this.initMemoryGame();
              }, 500);
            }
          } else {
            setTimeout(() => {
              c1.card.flipped = false;
              c2.card.flipped = false;
              c1.canvas.style.display = 'none';
              c1.cover.style.display = 'block';
              c2.canvas.style.display = 'none';
              c2.cover.style.display = 'block';
              this.memoryState.flippedCards = [];
            }, 1000);
          }
        }
      });

      gridEl.appendChild(cardEl);
    });
  }
}
