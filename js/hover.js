"use strict";

(function() {
  function createEl(template) {
    let el = document.createElement('div');
    el.innerHTML = template.trim();
    return el.firstChild;
  }

  function createSvgEl(template) {
    let el = createEl(`
      <svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">${template.trim()}</svg>
    `);
    return el;
  }

  function createSvgChildEl(template) {
    return createSvgEl(template).firstChild;
  }

  // Helper to set CSS properties (replaces dynamics.css)
  function setCss(el, props) {
    for (let key in props) {
      if (key === 'translateX') {
        el.style.transform = props[key] === 0 ? '' : `translateX(${props[key]}px)`;
      } else {
        el.style[key] = props[key];
      }
    }
  }

  let totalMaskIdx = 0;

  function createMasksWithStripes(count, box, averageHeight = 10) {
    let masks = [];
    for (let i = 0; i < count; i++) {
      masks.push([]);
    }
    let maskNames = [];
    for (let i = totalMaskIdx; i < totalMaskIdx + masks.length; i++) {
      maskNames.push(`clipPath${i}`);
    }
    totalMaskIdx += masks.length;
    let maskIdx = 0;
    let x = 0;
    let y = 0;
    let stripeHeight = averageHeight;
    while (true) {
      let w = Math.max(stripeHeight * 10, Math.round(Math.random() * box.width));
      masks[maskIdx].push(`
        M ${x},${y} L ${x + w},${y} L ${x + w},${y + stripeHeight} L ${x},${y + stripeHeight} Z
      `);

      maskIdx += 1;
      if (maskIdx >= masks.length) {
        maskIdx = 0;
      }

      x += w;
      if (x > box.width) {
        x = 0;
        y += stripeHeight;
        stripeHeight = Math.round(Math.random() * averageHeight + averageHeight / 2);
      }
      if (y >= box.height) {
        break;
      }
    }

    masks.forEach(function(rects, i) {
      let el = createSvgChildEl(`<clipPath id="${maskNames[i]}">
        <path d="${rects.join(' ')}" fill="white"></path>
      </clipPath>`);
      document.querySelector('#clip-paths g').appendChild(el);
    });

    return maskNames;
  }

  function cloneAndStripeElement(element, clipPathName, parent) {
    let el = element.cloneNode(true);
    let box = element.getBoundingClientRect();
    let parentBox = parent.getBoundingClientRect();
    box = {
      top: box.top - parentBox.top,
      left: box.left - parentBox.left,
      width: box.width,
      height: box.height,
    };
    let style = window.getComputedStyle(element);

    setCss(el, {
      position: 'absolute',
      left: Math.round(box.left + window.scrollX) + 'px',
      top: Math.round(box.top + window.scrollY) + 'px',
      width: Math.ceil(box.width) + 'px',
      height: Math.ceil(box.height) + 'px',
      display: 'none',
      pointerEvents: 'none',
      background: 'transparent',
      fontSize: style.fontSize,
      fontFamily: style.fontFamily,
      color: style.color,
      textDecoration: style.textDecoration,
    });
    parent.appendChild(el);
    el.style['-webkit-clip-path'] = `url(/#${clipPathName})`;
    el.style['clip-path'] = `url(/#${clipPathName})`;

    return el;
  }

  function handleMouseOver(e) {
    let el = e.target;
    while (el && el.tagName && el.tagName.toLowerCase() !== 'a') {
      el = el.parentNode;
    }
    if (!el || !el.tagName) {
      return;
    }
    let r = animateLink(el);

    let handleMouseOut = function(e) {
      el.removeEventListener('mouseout', handleMouseOut);
      r.stop();
    };

    el.addEventListener('mouseout', handleMouseOut);
  }

  function animateLink(el) {
    let animating = true;
    let box = el.getBoundingClientRect();

    let animate = function() {
      let masks = createMasksWithStripes(3, box, 3);
      let clonedEls = [];

      for (let i = 0; i < masks.length; i++) {
        let clonedEl = cloneAndStripeElement(el, masks[i], document.body);
        let childrenEls = Array.prototype.slice.apply(clonedEl.querySelectorAll('path'));
        childrenEls.push(clonedEl);
        for (let k = 0; k < childrenEls.length; k++) {
          let color = `hsl(${Math.round(Math.random() * 360)}, 80%, 65%)`;
          setCss(childrenEls[k], {
            color: color,
            fill: color,
          });
        }
        clonedEl.style.display = '';
        clonedEls.push(clonedEl);
      }

      for (let i = 0; i < clonedEls.length; i++) {
        let clonedEl = clonedEls[i];
        setCss(clonedEl, {
          translateX: Math.random() * 10 - 5,
        });

        setTimeout(function() {
          setCss(clonedEl, {
            translateX: 0,
          });
        }, 50);

        setTimeout(function() {
          setCss(clonedEl, {
            translateX: Math.random() * 5 - 2.5,
          });
        }, 100);

        setTimeout(function() {
          if (clonedEl.parentNode) {
            document.body.removeChild(clonedEl);
          }
        }, 150);
      }

      setTimeout(function() {
        if (animating) {
          animate();
        }
        for (let i = 0; i < masks.length; i++) {
          let maskEl = document.querySelector(`#${masks[i]}`);
          if (maskEl && maskEl.parentNode) {
            maskEl.parentNode.removeChild(maskEl);
          }
        }
      }, Math.random() * 1000);
    };

    animate();

    return {
      stop: function() {
        animating = false;
      },
    };
  }

  // Initialize hover effects on all links
  if (!('ontouchstart' in window)) {
    let linkEls = document.querySelectorAll('a');
    for (let i = 0; i < linkEls.length; i++) {
      linkEls[i].addEventListener('mouseover', handleMouseOver);
    }
  }
})();
