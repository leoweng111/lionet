const puppeteer = require('puppeteer');
(async () => {
  const browser = await puppeteer.launch({headless: false});
  const page = await browser.newPage();

  page.on('console', msg => {
    if (msg.text().indexOf('[vite]') === -1) console.log('PAGE LOG:', msg.text());
  });
  page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));

  await page.goto('http://localhost:5173/fusion');
  await new Promise(r => setTimeout(r, 2000));
  console.log('Now at:', page.url(), 'H2:', await page.evaluate(() => document.querySelector('h2')?.textContent));

  await page.evaluate(() => {
    const items = Array.from(document.querySelectorAll('.el-menu-item'));
    items.find(el => el.textContent.indexOf('因子挖掘') > -1)?.click();
  });
  await new Promise(r => setTimeout(r, 2000));
  console.log('Now at:', page.url(), 'H2:', await page.evaluate(() => document.querySelector('h2')?.textContent));

  await page.evaluate(() => {
    const items = Array.from(document.querySelectorAll('.el-menu-item'));
    items.find(el => el.textContent.indexOf('策略分析') > -1)?.click();
  });
  await new Promise(r => setTimeout(r, 2000));
  console.log('Now at:', page.url(), 'H2:', await page.evaluate(() => document.querySelector('h2')?.textContent));

  await browser.close();
})();
