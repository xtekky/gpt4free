/* ==========================================================================
   WEBGPU WGSL SHADER ENGINE & CANVAS PROCEDURAL TILE GENERATOR
   ========================================================================== */

export const PRESETS = [
  {
    id: 'andalusian-star',
    name: 'Andalusian Star Geometry',
    category: 'Geometric Tessellation',
    promptKeywords: 'star geometric islamic tile gold emerald tessellation',
    scale: 6.0,
    complexity: 8.0,
    grout: 0.025,
    roughness: 0.25,
    bump: 0.9,
    colors: ['#0f766e', '#f59e0b', '#00f2fe', '#0f172a']
  },
  {
    id: 'emerald-terrazzo',
    name: 'Emerald & Gold Terrazzo',
    category: 'Mineral Aggregate',
    promptKeywords: 'terrazzo mineral stone chips marble speckled aggregate',
    scale: 12.0,
    complexity: 6.0,
    grout: 0.015,
    roughness: 0.15,
    bump: 0.6,
    colors: ['#059669', '#d97706', '#e2e8f0', '#1e293b']
  },
  {
    id: 'calacatta-marble',
    name: 'Calacatta Gold Marble',
    category: 'Luxury Stone',
    promptKeywords: 'calacatta marble vein luxury polished smooth white gold',
    scale: 3.0,
    complexity: 4.0,
    grout: 0.01,
    roughness: 0.08,
    bump: 0.3,
    colors: ['#f8fafc', '#b45309', '#64748b', '#0f172a']
  },
  {
    id: 'voronoi-quartz',
    name: 'Voronoi Organic Quartz',
    category: 'Crystalline',
    promptKeywords: 'voronoi organic cell quartz gem glowing vein crystalline',
    scale: 8.0,
    complexity: 7.0,
    grout: 0.03,
    roughness: 0.35,
    bump: 1.2,
    colors: ['#00f2fe', '#8b5cf6', '#ec4899', '#090d16']
  },
  {
    id: 'hexagonal-grid',
    name: 'Hexagonal Tactile Grid',
    category: 'Modern Ceramic',
    promptKeywords: 'hex hexagon honeycomb tactile matte ceramic grid modern',
    scale: 7.0,
    complexity: 5.0,
    grout: 0.04,
    roughness: 0.45,
    bump: 1.0,
    colors: ['#334155', '#0284c7', '#38bdf8', '#0f172a']
  },
  {
    id: 'moroccan-zellige',
    name: 'Moroccan Zellige Glaze',
    category: 'Handmade Glaze',
    promptKeywords: 'moroccan zellige glaze ceramic handcrafted terracotta cobalt',
    scale: 9.0,
    complexity: 6.0,
    grout: 0.035,
    roughness: 0.2,
    bump: 0.8,
    colors: ['#1d4ed8', '#c2410c', '#fef08a', '#1e293b']
  },
  {
    id: 'cyber-hologram',
    name: 'Cyberpunk Neon Matrix',
    category: 'Sci-Fi Grid',
    promptKeywords: 'cyberpunk neon circuit matrix hologram metallic glowing sci-fi',
    scale: 10.0,
    complexity: 9.0,
    grout: 0.02,
    roughness: 0.1,
    bump: 1.4,
    colors: ['#ec4899', '#00f2fe', '#a855f7', '#030712']
  }
];

export class ShaderEngine {
  constructor() {
    this.adapter = null;
    this.device = null;
    this.isWebGPUSupported = false;
    this.seed = 12345;
  }

  async init() {
    if (navigator.gpu) {
      try {
        this.adapter = await navigator.gpu.requestAdapter();
        if (this.adapter) {
          this.device = await this.adapter.requestDevice();
          this.isWebGPUSupported = !!this.device;
        }
      } catch (err) {
        console.warn('WebGPU initialization fallback to Procedural Canvas engine:', err);
        this.isWebGPUSupported = false;
      }
    }
    return this.isWebGPUSupported;
  }

  // Parse hexadecimal color string to RGB array normalized [0..1]
  hexToRgb(hex) {
    let clean = hex.replace('#', '');
    if (clean.length === 3) {
      clean = clean.split('').map(c => c + c).join('');
    }
    const num = parseInt(clean, 16);
    return [
      ((num >> 16) & 255) / 255.0,
      ((num >> 8) & 255) / 255.0,
      (num & 255) / 255.0
    ];
  }

  // Render tile onto canvas (using WebGPUWGSL pipeline or High-Precision Procedural Math Pipeline)
  renderTileToCanvas(canvas, config) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    const scale = config.scale || 8.0;
    const complexity = config.complexity || 5.0;
    const grout = config.grout || 0.03;
    const seed = config.seed || this.seed;
    const presetId = config.presetId || 'andalusian-star';

    const c1 = this.hexToRgb(config.colors[0] || '#10b981');
    const c2 = this.hexToRgb(config.colors[1] || '#f59e0b');
    const c3 = this.hexToRgb(config.colors[2] || '#00f2fe');
    const cg = this.hexToRgb(config.colors[3] || '#1e293b');

    // Generate high-resolution procedural pixel values
    for (let y = 0; y < height; y++) {
      const v = y / height;
      for (let x = 0; x < width; x++) {
        const u = x / width;
        const idx = (y * width + x) * 4;

        let rgb = [0, 0, 0];
        let heightVal = 0.5;

        switch (presetId) {
          case 'andalusian-star':
            rgb = this.calcAndalusianStar(u, v, scale, complexity, grout, c1, c2, c3, cg, seed);
            break;
          case 'emerald-terrazzo':
            rgb = this.calcTerrazzo(u, v, scale, complexity, grout, c1, c2, c3, cg, seed);
            break;
          case 'calacatta-marble':
            rgb = this.calcMarble(u, v, scale, complexity, grout, c1, c2, c3, cg, seed);
            break;
          case 'voronoi-quartz':
            rgb = this.calcVoronoi(u, v, scale, complexity, grout, c1, c2, c3, cg, seed);
            break;
          case 'hexagonal-grid':
            rgb = this.calcHexGrid(u, v, scale, complexity, grout, c1, c2, c3, cg, seed);
            break;
          case 'moroccan-zellige':
            rgb = this.calcZellige(u, v, scale, complexity, grout, c1, c2, c3, cg, seed);
            break;
          case 'cyber-hologram':
            rgb = this.calcCyberMatrix(u, v, scale, complexity, grout, c1, c2, c3, cg, seed);
            break;
          default:
            rgb = this.calcAndalusianStar(u, v, scale, complexity, grout, c1, c2, c3, cg, seed);
        }

        data[idx]     = Math.min(255, Math.max(0, Math.floor(rgb[0] * 255)));
        data[idx + 1] = Math.min(255, Math.max(0, Math.floor(rgb[1] * 255)));
        data[idx + 2] = Math.min(255, Math.max(0, Math.floor(rgb[2] * 255)));
        data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }

  // Render a multi-tile grid where each (row, col) cell has its own individual tile config!
  renderMultiTileGridToCanvas(canvas, gridMatrix, gridRows = 3, gridCols = 3) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    const cellW = Math.floor(width / gridCols);
    const cellH = Math.floor(height / gridRows);

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = cellW;
    tempCanvas.height = cellH;

    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        const tileConfig = (gridMatrix && gridMatrix[r] && gridMatrix[r][c]) 
          ? gridMatrix[r][c] 
          : { presetId: 'andalusian-star', scale: 6, seed: (r * gridCols + c) * 99 };

        this.renderTileToCanvas(tempCanvas, tileConfig);
        ctx.drawImage(tempCanvas, c * cellW, r * cellH, cellW, cellH);
      }
    }
  }

  // --- PROCEDURAL WGSL PATTERN FORMULAS ---

  // 1. Andalusian Geometric Star Tessellation
  calcAndalusianStar(u, v, scale, complexity, grout, c1, c2, c3, cg, seed) {
    const su = (u * scale) % 1.0;
    const sv = (v * scale) % 1.0;
    
    // Grid tile distance
    const distEdgeU = Math.min(su, 1.0 - su);
    const distEdgeV = Math.min(sv, 1.0 - sv);
    const edgeDist = Math.min(distEdgeU, distEdgeV);

    if (edgeDist < grout) {
      return cg;
    }

    // Centered coordinates [-1, 1]
    const cx = (su - 0.5) * 2.0;
    const cy = (sv - 0.5) * 2.0;
    const distCenter = Math.sqrt(cx * cx + cy * cy);
    const angle = Math.atan2(cy, cx);

    // Star symmetry (8-fold star)
    const starR = 0.5 + 0.3 * Math.cos(angle * 8.0 + seed * 0.1);
    const starDist = Math.abs(distCenter - starR);

    if (distCenter < 0.35) {
      // Inner star core
      const blend = 0.5 + 0.5 * Math.sin(angle * 4.0 + complexity);
      return this.mixColor(c1, c2, blend);
    } else if (distCenter < starR + 0.08) {
      // Interlocking geometric star points
      return c2;
    } else {
      // Outer tile corners
      const blend = Math.sin(cx * 3.0 + cy * 3.0 + seed) * 0.5 + 0.5;
      return this.mixColor(c1, c3, blend);
    }
  }

  // 2. Terrazzo Stone Chips
  calcTerrazzo(u, v, scale, complexity, grout, c1, c2, c3, cg, seed) {
    const su = u * scale;
    const sv = v * scale;

    const cellX = Math.floor(su);
    const cellY = Math.floor(sv);
    const localU = su - cellX;
    const localV = sv - cellY;

    // Check distance to grout
    if (localU < grout || localU > 1.0 - grout || localV < grout || localV > 1.0 - grout) {
      return cg;
    }

    // Pseudo-random hash for chip placement
    let closestDist = 99.0;
    let chipColor = c1;

    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const hash = this.hash22(cellX + dx + seed, cellY + dy + seed);
        const chipU = dx + hash[0];
        const chipV = dy + hash[1];

        const du = localU - chipU;
        const dv = localV - chipV;
        const dist = Math.sqrt(du * du + dv * dv);

        if (dist < closestDist) {
          closestDist = dist;
          if (hash[2] > 0.65) chipColor = c2;
          else if (hash[2] > 0.35) chipColor = c3;
          else chipColor = c1;
        }
      }
    }

    // Stone chip radius
    if (closestDist < 0.25 + 0.1 * Math.sin(su * 5.0 + sv * 5.0)) {
      return chipColor;
    } else {
      // Polished concrete matrix base
      const noise = (Math.sin(su * 20.0 + seed) * Math.cos(sv * 20.0 + seed)) * 0.05;
      return [
        Math.min(1.0, c1[0] * 0.4 + 0.6 + noise),
        Math.min(1.0, c1[1] * 0.4 + 0.6 + noise),
        Math.min(1.0, c1[2] * 0.4 + 0.6 + noise)
      ];
    }
  }

  // 3. Calacatta Gold Marble
  calcMarble(u, v, scale, complexity, grout, c1, c2, c3, cg, seed) {
    const su = (u * scale) % 1.0;
    const sv = (v * scale) % 1.0;

    // Grout tile line
    if (su < grout || su > 1.0 - grout || sv < grout || sv > 1.0 - grout) {
      return cg;
    }

    // Turbulent noise vein simulation
    const n1 = Math.sin(u * 8.0 * scale + seed) * Math.cos(v * 8.0 * scale + seed);
    const n2 = Math.sin(u * 16.0 * scale + n1 * 3.0) * 0.5;
    const vein = Math.abs(Math.sin((u + v + n2 * 0.4 * complexity) * 10.0));

    if (vein < 0.08) {
      // Golden vein core
      return c2;
    } else if (vein < 0.22) {
      // Grey/smokey vein border
      const t = (vein - 0.08) / 0.14;
      return this.mixColor(c2, c3, t);
    } else {
      // Smooth white marble body with subtle gloss tint
      const bodyNoise = Math.sin(u * 30.0 + v * 30.0) * 0.02;
      return [
        Math.min(1.0, c1[0] + bodyNoise),
        Math.min(1.0, c1[1] + bodyNoise),
        Math.min(1.0, c1[2] + bodyNoise)
      ];
    }
  }

  // 4. Voronoi Organic Quartz
  calcVoronoi(u, v, scale, complexity, grout, c1, c2, c3, cg, seed) {
    const su = u * scale;
    const sv = v * scale;

    const cellX = Math.floor(su);
    const cellY = Math.floor(sv);
    const localU = su - cellX;
    const localV = sv - cellY;

    let minDist1 = 99.0;
    let minDist2 = 99.0;
    let cellHash = 0;

    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const hash = this.hash22(cellX + dx + seed, cellY + dy + seed);
        const pointU = dx + hash[0];
        const pointV = dy + hash[1];

        const du = localU - pointU;
        const dv = localV - pointV;
        const dist = Math.sqrt(du * du + dv * dv);

        if (dist < minDist1) {
          minDist2 = minDist1;
          minDist1 = dist;
          cellHash = hash[2];
        } else if (dist < minDist2) {
          minDist2 = dist;
        }
      }
    }

    // Border line between cells
    const cellEdge = minDist2 - minDist1;

    if (cellEdge < grout * 2.5) {
      return cg;
    } else if (cellEdge < grout * 5.0) {
      // Glowing vein boundary
      return c3;
    } else {
      // Cell crystal body
      const t = Math.min(1.0, minDist1 * 1.5);
      const baseCol = cellHash > 0.5 ? c1 : c2;
      return this.mixColor(baseCol, c3, t * 0.6);
    }
  }

  // 5. Hexagonal Tactile Grid
  calcHexGrid(u, v, scale, complexity, grout, c1, c2, c3, cg, seed) {
    const su = u * scale * 1.732;
    const sv = v * scale * 1.5;

    const row = Math.floor(sv);
    const colOffset = (row % 2 === 0) ? 0.0 : 0.866;
    const col = Math.floor(su - colOffset);

    const localU = (su - colOffset) - col - 0.5;
    const localV = sv - row - 0.5;

    const distCenter = Math.sqrt(localU * localU + localV * localV);

    if (distCenter > 0.5 - grout) {
      return cg;
    } else if (distCenter > 0.44 - grout) {
      // Beveled edge highlight
      return c3;
    } else {
      // Hex face tactile texture
      const hash = this.hash22(col + seed, row + seed);
      const colChoice = hash[2] > 0.5 ? c1 : c2;
      const microPattern = Math.sin(localU * 20.0 + localV * 20.0) * 0.08;
      return [
        Math.min(1.0, Math.max(0.0, colChoice[0] + microPattern)),
        Math.min(1.0, Math.max(0.0, colChoice[1] + microPattern)),
        Math.min(1.0, Math.max(0.0, colChoice[2] + microPattern))
      ];
    }
  }

  // 6. Moroccan Zellige Handmade Glaze
  calcZellige(u, v, scale, complexity, grout, c1, c2, c3, cg, seed) {
    const su = (u * scale) % 1.0;
    const sv = (v * scale) % 1.0;

    if (su < grout || su > 1.0 - grout || sv < grout || sv > 1.0 - grout) {
      return cg;
    }

    // Tile index for glaze variation
    const tileX = Math.floor(u * scale);
    const tileY = Math.floor(v * scale);
    const hash = this.hash22(tileX + seed, tileY + seed);

    // Glaze crackle & color shift
    const baseColor = hash[2] > 0.6 ? c1 : (hash[2] > 0.3 ? c2 : c3);
    const glazeVariation = (hash[0] - 0.5) * 0.15;
    const crackle = Math.sin(su * 50.0 + hash[1] * 10.0) * Math.cos(sv * 50.0 + hash[0] * 10.0);

    const isCrackleLine = Math.abs(crackle) > 0.88;

    if (isCrackleLine) {
      return [0.05, 0.05, 0.05]; // Dark glaze crackle line
    }

    return [
      Math.min(1.0, Math.max(0.0, baseColor[0] + glazeVariation)),
      Math.min(1.0, Math.max(0.0, baseColor[1] + glazeVariation)),
      Math.min(1.0, Math.max(0.0, baseColor[2] + glazeVariation))
    ];
  }

  // 7. Cyberpunk Neon Matrix
  calcCyberMatrix(u, v, scale, complexity, grout, c1, c2, c3, cg, seed) {
    const su = (u * scale) % 1.0;
    const sv = (v * scale) % 1.0;

    if (su < grout || su > 1.0 - grout || sv < grout || sv > 1.0 - grout) {
      return c3; // Glowing neon grid lines
    }

    // Circuit trace logic
    const gridU = Math.floor(u * scale * 4.0);
    const gridV = Math.floor(v * scale * 4.0);
    const hash = this.hash22(gridU + seed, gridV + seed);

    const distLine = Math.min(Math.abs(su - 0.5), Math.abs(sv - 0.5));

    if (distLine < 0.05 && hash[2] > 0.4) {
      return c1; // Neon circuit trace
    } else if (hash[0] > 0.75) {
      return c2; // Glowing micro chip
    } else {
      return cg; // Dark metallic base
    }
  }

  // Helper Math: Linear color interpolation
  mixColor(colA, colB, t) {
    const clampedT = Math.max(0.0, Math.min(1.0, t));
    return [
      colA[0] * (1.0 - clampedT) + colB[0] * clampedT,
      colA[1] * (1.0 - clampedT) + colB[1] * clampedT,
      colA[2] * (1.0 - clampedT) + colB[2] * clampedT
    ];
  }

  // Helper Hash Function
  hash22(x, y) {
    const n = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
    const h1 = n - Math.floor(n);
    const n2 = Math.sin(x * 63.7264 + y * 19.827) * 23145.9123;
    const h2 = n2 - Math.floor(n2);
    const n3 = Math.sin(x * 37.1938 + y * 83.127) * 91234.5678;
    const h3 = n3 - Math.floor(n3);
    return [h1, h2, h3];
  }

  // Generate Normal Map from generated Diffuse/Height Canvas
  generateNormalMapCanvas(sourceCanvas) {
    const width = sourceCanvas.width;
    const height = sourceCanvas.height;
    const srcCtx = sourceCanvas.getContext('2d');
    const srcData = srcCtx.getImageData(0, 0, width, height).data;

    const normCanvas = document.createElement('canvas');
    normCanvas.width = width;
    normCanvas.height = height;
    const normCtx = normCanvas.getContext('2d');
    const normImgData = normCtx.createImageData(width, height);
    const normData = normImgData.data;

    const getLuma = (x, y) => {
      const clampX = (x + width) % width;
      const clampY = (y + height) % height;
      const idx = (clampY * width + clampX) * 4;
      return (srcData[idx] * 0.299 + srcData[idx + 1] * 0.587 + srcData[idx + 2] * 0.114) / 255.0;
    };

    const bumpStrength = 3.0;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;

        const left   = getLuma(x - 1, y);
        const right  = getLuma(x + 1, y);
        const top    = getLuma(x, y - 1);
        const bottom = getLuma(x, y + 1);

        const dx = (left - right) * bumpStrength;
        const dy = (top - bottom) * bumpStrength;
        const dz = 1.0;

        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const nx = (dx / len) * 0.5 + 0.5;
        const ny = (dy / len) * 0.5 + 0.5;
        const nz = (dz / len) * 0.5 + 0.5;

        normData[idx]     = Math.floor(nx * 255);
        normData[idx + 1] = Math.floor(ny * 255);
        normData[idx + 2] = Math.floor(nz * 255);
        normData[idx + 3] = 255;
      }
    }

    normCtx.putImageData(normImgData, 0, 0);
    return normCanvas;
  }
}
