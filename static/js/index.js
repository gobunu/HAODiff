window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

})

// 添加返回顶部按钮的功能
document.addEventListener('DOMContentLoaded', function() {
  // 获取返回顶部按钮
  const backToTopButton = document.getElementById('back-to-top');
  
  // 监听滚动事件
  window.addEventListener('scroll', function() {
    if (window.pageYOffset > 300) { // 当页面滚动超过300px时显示按钮
      backToTopButton.style.display = 'block';
    } else {
      backToTopButton.style.display = 'none';
    }
  });
  
  // 点击按钮返回顶部
  backToTopButton.addEventListener('click', function() {
    window.scrollTo({
      top: 0,
      behavior: 'smooth' // 平滑滚动
    });
  });
});

// 图片加载优化
document.addEventListener('DOMContentLoaded', function() {
  const images = document.querySelectorAll('img');
  images.forEach(img => {
    img.addEventListener('load', function() {
      this.style.opacity = 1;
    });
  });
});

// 图片懒加载功能
document.addEventListener('DOMContentLoaded', function() {
  const lazyImages = document.querySelectorAll('img[data-src]');
  
  const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        img.src = img.dataset.src; // 将 data-src 的值赋给 src
        img.classList.add('loaded');
        observer.unobserve(img); // 加载完成后取消观察
      }
    });
  }, {
    rootMargin: '50px 0px' // 提前50px开始加载
  });

  lazyImages.forEach(img => {
    imageObserver.observe(img);
  });
});

// Image Comparer functionality
document.addEventListener('DOMContentLoaded', function() {
  function initImageComparer(element) {
    const before = element.querySelector('.image-before');
    const after = element.querySelector('.image-after');
    const sliderLine = element.querySelector('.slider-line');
    let isActive = false;

    // Load images from data attributes
    const beforeSrc = element.getAttribute('data-before');
    const afterSrc = element.getAttribute('data-after');
    
    if (beforeSrc) before.src = beforeSrc;
    if (afterSrc) after.src = afterSrc;

    function updateSlider(x) {
      const rect = element.getBoundingClientRect();
      const position = Math.max(0, Math.min(rect.width, x - rect.left));
      
      sliderLine.style.left = position + 'px';
      before.style.clip = `rect(0, ${position}px, ${rect.height}px, 0)`;
    }

    // Mouse events
    element.addEventListener('mousedown', (e) => {
      isActive = true;
      updateSlider(e.clientX);
      e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
      if (!isActive) return;
      updateSlider(e.clientX);
    });

    document.addEventListener('mouseup', () => {
      isActive = false;
    });

    // Touch events for mobile
    element.addEventListener('touchstart', (e) => {
      isActive = true;
      updateSlider(e.touches[0].clientX);
      e.preventDefault();
    });

    document.addEventListener('touchmove', (e) => {
      if (!isActive) return;
      updateSlider(e.touches[0].clientX);
      e.preventDefault();
    }, { passive: false });

    document.addEventListener('touchend', () => {
      isActive = false;
    });

    // Click to move slider
    element.addEventListener('click', (e) => {
      if (!isActive) {
        updateSlider(e.clientX);
      }
    });
  }

  // Initialize all image comparers
  const comparers = document.querySelectorAll('.image-comparer');
  comparers.forEach(initImageComparer);
});

// Hero particles background
document.addEventListener('DOMContentLoaded', function () {
  const canvas = document.getElementById('hero-particles');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let width, height, particles;

  function resize() {
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
  }

  function createParticles() {
    const count = 60;
    particles = new Array(count).fill(0).map(() => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: 1.2 + Math.random() * 1.8
    }));
  }

  function step() {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(148, 163, 184, 0.8)';

    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > width) p.vx *= -1;
      if (p.y < 0 || p.y > height) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.strokeStyle = 'rgba(94, 234, 212, 0.35)';
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist2 = dx * dx + dy * dy;
        if (dist2 < 160 * 160) {
          const alpha = 1 - dist2 / (160 * 160);
          ctx.globalAlpha = alpha;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }
    }

    requestAnimationFrame(step);
  }

  resize();
  createParticles();
  step();
  window.addEventListener('resize', function () {
    resize();
    createParticles();
  });
});

// Scroll progress bar
document.addEventListener('DOMContentLoaded', function () {
  const bar = document.getElementById('scroll-progress');
  if (!bar) return;

  function update() {
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = docHeight > 0 ? (scrollTop / docHeight) : 0;
    bar.style.width = (ratio * 100) + '%';
  }

  window.addEventListener('scroll', update, { passive: true });
  update();
});

// Theme toggle
document.addEventListener('DOMContentLoaded', function () {
  const btn = document.getElementById('theme-toggle');
  if (!btn) return;

  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const stored = localStorage.getItem('haodiff-theme');
  if (stored === 'dark' || (!stored && prefersDark)) {
    document.body.classList.add('dark-theme');
  }

  function renderIcon() {
    const i = btn.querySelector('i');
    if (!i) return;
    if (document.body.classList.contains('dark-theme')) {
      i.classList.remove('fa-moon');
      i.classList.add('fa-sun');
    } else {
      i.classList.remove('fa-sun');
      i.classList.add('fa-moon');
    }
  }

  renderIcon();

  btn.addEventListener('click', function () {
    const isDark = document.body.classList.toggle('dark-theme');
    localStorage.setItem('haodiff-theme', isDark ? 'dark' : 'light');
    renderIcon();
  });
});

// Image lightbox / gallery for result images
document.addEventListener('DOMContentLoaded', function () {
  const resultSection = document.querySelector('section .hero-body') || document;
  const candidates = document.querySelectorAll("img[data-src^='static/images/fig'], img[data-src^='static/images/tab']");
  if (!candidates.length) return;

  const overlay = document.createElement('div');
  overlay.className = 'lightbox-overlay';
  overlay.innerHTML = `
    <div class="lightbox-content">
      <button class="lightbox-close">&times;</button>
      <button class="lightbox-prev">&#8249;</button>
      <div class="lightbox-image-wrapper">
        <img class="lightbox-image" src="" alt="result preview" />
      </div>
      <button class="lightbox-next">&#8250;</button>
      <div class="lightbox-zoom-hint">Scroll / dblclick to zoom</div>
    </div>
  `;
  document.body.appendChild(overlay);

  const imgEl = overlay.querySelector('.lightbox-image');
  const btnClose = overlay.querySelector('.lightbox-close');
  const btnPrev = overlay.querySelector('.lightbox-prev');
  const btnNext = overlay.querySelector('.lightbox-next');
  const wrapper = overlay.querySelector('.lightbox-image-wrapper');

  const images = Array.from(candidates);
  let index = 0;

  function show(idx) {
    index = (idx + images.length) % images.length;
    const el = images[index];
    const src = el.getAttribute('data-src') || el.src;
    wrapper.style.transform = 'translate(0px, 0px) scale(1)';
    imgEl.src = src;
    overlay.classList.add('active');
  }

  images.forEach((el, i) => {
    el.style.cursor = 'zoom-in';
    el.addEventListener('click', function () {
      show(i);
    });
  });

  function hide() {
    overlay.classList.remove('active');
  }

  btnClose.addEventListener('click', hide);
  overlay.addEventListener('click', function (e) {
    if (e.target === overlay) hide();
  });
  btnPrev.addEventListener('click', function (e) {
    e.stopPropagation();
    show(index - 1);
  });
  btnNext.addEventListener('click', function (e) {
    e.stopPropagation();
    show(index + 1);
  });

  document.addEventListener('keydown', function (e) {
    if (!overlay.classList.contains('active')) return;
    if (e.key === 'Escape') hide();
    if (e.key === 'ArrowLeft') show(index - 1);
    if (e.key === 'ArrowRight') show(index + 1);
  });
  // zoom & pan
  let zoom = 1;
  let offsetX = 0;
  let offsetY = 0;
  let isPanning = false;
  let startX = 0;
  let startY = 0;

  function applyTransform() {
    wrapper.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${zoom})`;
  }

  overlay.addEventListener('wheel', function (e) {
    if (!overlay.classList.contains('active')) return;
    e.preventDefault();
    const delta = -e.deltaY;
    const factor = delta > 0 ? 1.1 : 0.9;
    const newZoom = Math.min(4, Math.max(1, zoom * factor));
    zoom = newZoom;
    applyTransform();
  }, { passive: false });

  imgEl.addEventListener('dblclick', function (e) {
    e.stopPropagation();
    if (zoom === 1) {
      zoom = 2;
    } else {
      zoom = 1;
      offsetX = 0;
      offsetY = 0;
    }
    applyTransform();
  });

  wrapper.addEventListener('mousedown', function (e) {
    if (zoom === 1) return;
    isPanning = true;
    startX = e.clientX - offsetX;
    startY = e.clientY - offsetY;
    e.preventDefault();
  });

  document.addEventListener('mousemove', function (e) {
    if (!isPanning) return;
    offsetX = e.clientX - startX;
    offsetY = e.clientY - startY;
    applyTransform();
  });

  document.addEventListener('mouseup', function () {
    isPanning = false;
  });
});

// Auto-dark by time (00:00 - 06:00) in addition to preferences
document.addEventListener('DOMContentLoaded', function () {
  const btn = document.getElementById('theme-toggle');
  if (!btn) return;
  const stored = localStorage.getItem('haodiff-theme');
  if (!stored) {
    const h = new Date().getHours();
    if (h >= 0 && h < 6) {
      document.body.classList.add('dark-theme');
    }
  }
});

// Initial auto-swipe for image comparers
document.addEventListener('DOMContentLoaded', function () {
  const comparers = document.querySelectorAll('.image-comparer');
  // Skip auto animation on small screens / touch-centric devices
  const isSmallScreen = window.matchMedia && window.matchMedia('(max-width: 768px)').matches;
  if (isSmallScreen) return;
  comparers.forEach(function (el, idx) {
    const before = el.querySelector('.image-before');
    const sliderLine = el.querySelector('.slider-line');
    const rect = el.getBoundingClientRect();
    const start = rect.width * 0.9;
    const end = rect.width * 0.1;
    let startTime = null;

    function animate(ts) {
      if (!startTime) startTime = ts;
      const progress = Math.min(1, (ts - startTime) / 2000); // 2s slow swipe
      const pos = start + (end - start) * progress;
      sliderLine.style.left = pos + 'px';
      before.style.clip = `rect(0, ${pos}px, ${rect.height}px, 0)`;
      if (progress < 1) requestAnimationFrame(animate);
    }

    // stagger animations a bit
    setTimeout(function () {
      requestAnimationFrame(animate);
    }, idx * 300);
  });
});

// Cursor trail disabled (kept SVG in DOM but no drawing logic)
