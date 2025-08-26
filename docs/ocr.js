/* GitHub Pages build of the client. Same logic as client/ocr.js but loads locally from docs/. */
// API base URL; override on static hosting (e.g., GitHub Pages) by setting window.OCR_API_URL
const SERVER_URL = ((window.OCR_API_URL || window.location.origin) + '').replace(/\/$/, '');

// Canvas and drawing settings
const DOWNSCALE = 28; // target 28x28 for OCR input
const BRUSH_SIZE = 18; // base brush width in CSS pixels; scaled by devicePixelRatio

// Canvas setup
theCanvas();
function theCanvas(){
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const clearBtn = document.getElementById('clearBtn');
  const predictBtn = document.getElementById('predictBtn');
  const trainBtn = document.getElementById('trainBtn');
  const labelInput = document.getElementById('labelInput');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  const brandToggle = document.getElementById('brandToggle');
  const dropdown = document.getElementById('dropdown');

  let drawing = false;
  let dpr = Math.max(1, window.devicePixelRatio || 1);

  function resizeCanvas(){
    const cssWidth = window.innerWidth;
    const cssHeight = window.innerHeight;
    dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.floor(cssWidth * dpr);
    canvas.height = Math.floor(cssHeight * dpr);
    ctx.setTransform(1,0,0,1,0,0);
    ctx.fillStyle = 'white';
    ctx.fillRect(0,0,canvas.width,canvas.height);
    drawGrid();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'black';
    ctx.lineWidth = BRUSH_SIZE * dpr;
  }

  function getPos(evt){
    const rect = canvas.getBoundingClientRect();
    const clientX = (evt.touches ? evt.touches[0].clientX : evt.clientX);
    const clientY = (evt.touches ? evt.touches[0].clientY : evt.clientY);
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return { x: (clientX - rect.left) * scaleX, y: (clientY - rect.top) * scaleY };
  }

  function startDraw(evt){ drawing = true; const {x,y} = getPos(evt); ctx.beginPath(); ctx.moveTo(x,y); evt.preventDefault(); }
  function draw(evt){ if(!drawing) return; const {x,y} = getPos(evt); ctx.lineTo(x,y); ctx.stroke(); evt.preventDefault(); }
  function endDraw(evt){ drawing = false; evt && evt.preventDefault(); }

  function clearCanvas(){
    ctx.save(); ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,canvas.width,canvas.height); ctx.restore();
    ctx.fillStyle='white'; ctx.fillRect(0,0,canvas.width,canvas.height); drawGrid(); resultEl.textContent='';
  }

  function getImageArray(){
    const off = document.createElement('canvas');
    off.width = DOWNSCALE; off.height = DOWNSCALE;
    const octx = off.getContext('2d');
    octx.fillStyle='white'; octx.fillRect(0,0,off.width,off.height);
    octx.drawImage(canvas,0,0,canvas.width,canvas.height,0,0,off.width,off.height);
    const img = octx.getImageData(0,0,off.width,off.height);
    const data = img.data; const arr = new Array(DOWNSCALE*DOWNSCALE);
    for(let i=0,j=0;i<data.length;i+=4,j++){
      const r=data[i],g=data[i+1],b=data[i+2];
      const gray = 0.299*r + 0.587*g + 0.114*b; const v = 1 - gray/255; arr[j] = +v.toFixed(4);
    }
    return arr;
  }

  function drawGrid(){
    const w=canvas.width,h=canvas.height,g=28; const stepX=w/g,stepY=h/g;
    ctx.save(); ctx.lineWidth = 1*dpr; ctx.strokeStyle='rgba(0,0,0,0.08)'; ctx.beginPath();
    for(let i=1;i<g;i++){ const x=Math.round(i*stepX)+0.5; ctx.moveTo(x,0); ctx.lineTo(x,h); }
    for(let j=1;j<g;j++){ const y=Math.round(j*stepY)+0.5; ctx.moveTo(0,y); ctx.lineTo(w,y); }
    ctx.stroke(); ctx.strokeStyle='rgba(0,0,0,0.16)'; ctx.beginPath();
    const major=7; for(let i=major;i<g;i+=major){ const x=Math.round(i*stepX)+0.5; ctx.moveTo(x,0); ctx.lineTo(x,h); }
    for(let j=major;j<g;j+=major){ const y=Math.round(j*stepY)+0.5; ctx.moveTo(0,y); ctx.lineTo(w,y); }
    ctx.stroke(); ctx.restore();
  }

  async function predict(){
    const image = getImageArray(); statusEl.textContent='Predicting…';
    try{ const res = await fetch(`${SERVER_URL}/api/predict`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image})});
      if(!res.ok) throw new Error(`HTTP ${res.status}`); const data = await res.json();
      resultEl.textContent = `Prediction: ${data.digit}  (confidence: ${(data.confidence*100).toFixed(1)}%)`;
    }catch(err){ resultEl.textContent = `Error: ${err.message}`; } finally { statusEl.textContent=''; }
  }

  async function train(){
    const label = Number(labelInput.value);
    if(!Number.isInteger(label) || label<0 || label>9){ resultEl.textContent='Enter a label 0-9 to train.'; return; }
    const image = getImageArray(); statusEl.textContent='Sending training sample…';
    try{ const res = await fetch(`${SERVER_URL}/api/train`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image,label})});
      if(!res.ok) throw new Error(`HTTP ${res.status}`); const data = await res.json();
      resultEl.textContent = data.message || 'Training sample stored.';
    }catch(err){ resultEl.textContent = `Error: ${err.message}`; } finally { statusEl.textContent=''; }
  }

  function bindEvents(){
    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', endDraw);
    canvas.addEventListener('touchstart', startDraw,{passive:false});
    canvas.addEventListener('touchmove', draw,{passive:false});
    window.addEventListener('touchend', endDraw,{passive:false});

    clearBtn.addEventListener('click', clearCanvas);
    predictBtn.addEventListener('click', predict);
    trainBtn.addEventListener('click', train);
    window.addEventListener('resize', resizeCanvas);

    // Prevent accidental scroll/arrow increments
    labelInput.addEventListener('wheel', (e)=>e.preventDefault(), {passive:false});
    labelInput.addEventListener('keydown', (e)=>{ const blocked=['ArrowUp','ArrowDown','PageUp','PageDown']; if(blocked.includes(e.key)&&!e.ctrlKey&&!e.metaKey) e.preventDefault(); });

    // Close dropdown on outside click
    brandToggle?.addEventListener('click', ()=>{
      const isOpen = dropdown.hasAttribute('hidden') ? false : true;
      if(isOpen){ dropdown.setAttribute('hidden',''); brandToggle.setAttribute('aria-expanded','false'); }
      else { dropdown.removeAttribute('hidden'); brandToggle.setAttribute('aria-expanded','true'); }
    });
    document.addEventListener('click', (e)=>{
      if(!dropdown) return; const btn=brandToggle; const tgt=e.target;
      if(!btn.contains(tgt) && !dropdown.contains(tgt)){
        if(!dropdown.hasAttribute('hidden')){ dropdown.setAttribute('hidden',''); brandToggle.setAttribute('aria-expanded','false'); }
      }
    });
  }

  resizeCanvas();
  bindEvents();
}
