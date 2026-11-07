'use strict';
const select = document.querySelector('#page-select');
const box = document.querySelector('#reader-pages');
const previous = document.querySelector('#prev-page');
const next = document.querySelector('#next-page');
function go(pageNumber) {
  const number = Math.max(1, Math.min(select.options.length, pageNumber));
  select.value = String(number);
  const page = document.querySelector('#page-' + number);
  const top = page.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - 24;
  box.scrollTo({ top, behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  previous.disabled = number === 1;
  next.disabled = number === select.options.length;
}
select.addEventListener('change', () => go(Number(select.value)));
previous.addEventListener('click', () => go(Number(select.value) - 1));
next.addEventListener('click', () => go(Number(select.value) + 1));
previous.disabled = true;
