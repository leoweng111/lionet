const puppeteer = require('puppeteer');
(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  page.on('console', msg => {
    if (msg.text().indexOf('[vite]') === -1) console.log('PAGE LOG:', msg.text());
  });
  page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));
  await page.goto('http://localhost:5173/mining');
  await new Promise(r => setTimeout(r, 2000));
  console.log('URL1:', page.url());
  await page.evaluate(() => {
    const items = Array.from(document.querySelectorAll('.el-menu-item'));
    items.find(el => el.textContent.indexOf('因子融合') > -1)?.click();
  });
  await new Promise(r => setTimeout(r, 1000));
  console.log('URL2:', page.url());
  console.log('H2:', await page.evaluate(() => document.querySelector('h2')?.textContent));
  await browser.close();
})();

