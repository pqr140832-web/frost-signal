// Frostline service worker: push notifications
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
// 页面本身先走网络，失败时用上次存的副本，这样 Pages 挂了也能打开并切到备用库
const SHELL = 'frostline-shell-v1';
self.addEventListener('fetch', event => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== self.location.origin) return;
  if (url.pathname.startsWith('/rest/') || url.pathname.startsWith('/storage/')) return;
  event.respondWith((async () => {
    try {
      const res = await fetch(req);
      if (res.ok) {
        const copy = res.clone();
        caches.open(SHELL).then(c => c.put(req, copy));
      }
      return res;
    } catch (e) {
      const hit = await caches.match(req, { ignoreSearch: req.mode === 'navigate' });
      if (hit) return hit;
      throw e;
    }
  })());
});

self.addEventListener('push', event => {
  let data = {};
  try { data = event.data ? event.data.json() : {}; } catch (e) { data = { title: '霜信', body: event.data ? event.data.text() : '' }; }
  event.waitUntil((async () => {
    await self.registration.showNotification(data.title || '新消息', {
      body: data.body || '',
      tag: data.conv_id || 'frostline',
      renotify: true,
      icon: '/icons/icon-192.png',
      badge: '/icons/badge-96.png',
      data: { conv_id: data.conv_id }
    });
  })());
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  const convId = event.notification.data && event.notification.data.conv_id;
  event.waitUntil((async () => {
    const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    for (const c of clients) {
      if ('focus' in c) {
        await c.focus();
        if (convId) c.postMessage({ type: 'open-conv', conv_id: convId });
        return;
      }
    }
    await self.clients.openWindow(convId ? '/?c=' + convId : '/');
  })());
});
