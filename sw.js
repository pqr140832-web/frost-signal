// Frostline service worker: push notifications
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', () => {});

self.addEventListener('push', event => {
  let data = {};
  try { data = event.data ? event.data.json() : {}; } catch (e) { data = { title: '霜信', body: event.data ? event.data.text() : '' }; }
  event.waitUntil((async () => {
    const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    if (clients.some(c => c.visibilityState === 'visible' && c.focused)) return;
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
