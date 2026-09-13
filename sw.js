// Frostline service worker: push notifications
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', () => {});

self.addEventListener('push', event => {
  let data = {};
  try { data = event.data ? event.data.json() : {}; } catch (e) { data = { title: '霜信', body: event.data ? event.data.text() : '' }; }
  event.waitUntil((async () => {
    const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    const visible = clients.some(c => c.visibilityState === 'visible');
    await self.registration.showNotification(data.title || '新消息', {
      body: data.body || '',
      tag: data.conv_id || 'frostline',
      renotify: true,
      icon: '/icons/icon-192.png',
      badge: '/icons/badge-96.png',
      data: { conv_id: data.conv_id },
      silent: visible
    });
    // 在 app 里时不打扰：iOS 要求每次推送都显示通知，否则会吊销订阅，所以显示后立刻关掉
    if (visible) {
      const list = await self.registration.getNotifications({ tag: data.conv_id || 'frostline' });
      list.forEach(n => n.close());
    }
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
