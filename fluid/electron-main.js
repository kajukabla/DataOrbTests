const { app, BrowserWindow, session, systemPreferences } = require('electron');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

let mainWindow;
let serverProcess;
const PORT = 8081;

const fluidRoot = app.isPackaged
  ? path.join(process.resourcesPath, 'app')
  : __dirname;

function isPortInUse() {
  return new Promise((resolve) => {
    const req = http.get(`http://localhost:${PORT}/`, () => resolve(true));
    req.on('error', () => resolve(false));
    req.setTimeout(1000, () => { req.destroy(); resolve(false); });
  });
}

function startServer() {
  return new Promise((resolve) => {
    const serverPath = path.join(fluidRoot, 'server.js');
    serverProcess = spawn(process.execPath, [serverPath], {
      cwd: fluidRoot,
      env: { ...process.env, ELECTRON_RUN_AS_NODE: '1', FLUID_AUTO_OPEN_BROWSER: '0' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    serverProcess.stdout?.on('data', (data) => {
      const msg = data.toString();
      console.log('[server]', msg.trim());
      if (msg.includes('Server running')) resolve();
    });

    serverProcess.stderr?.on('data', (data) => {
      const msg = data.toString();
      console.error('[server]', msg.trim());
      if (msg.includes('EADDRINUSE')) {
        serverProcess = null;
        resolve();
      }
    });

    serverProcess.on('error', (err) => {
      console.error('[server] spawn error:', err.message);
      serverProcess = null;
      resolve();
    });

    setTimeout(resolve, 3000);
  });
}

function setupPermissions() {
  // Auto-grant camera/mic permission requests from any window
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    console.log('[permissions] requested:', permission);
    if (permission === 'media' || permission === 'mediaKeySystem') {
      callback(true);
    } else {
      callback(true); // grant all for now
    }
  });
  session.defaultSession.setPermissionCheckHandler(() => true);
  session.defaultSession.setDevicePermissionHandler(() => true);
}

function showCameraPicker() {
  return new Promise((resolve) => {
    const picker = new BrowserWindow({
      width: 400,
      height: 480,
      title: 'DataOrb — Select Camera',
      icon: path.join(fluidRoot, 'build', 'icon.icns'),
      resizable: false,
      minimizable: false,
      maximizable: false,
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
      },
      backgroundColor: '#111111',
      titleBarStyle: 'hiddenInset',
      trafficLightPosition: { x: 12, y: 12 },
    });

    // Log renderer console to terminal
    picker.webContents.on('console-message', (e, level, msg) => {
      console.log('[picker]', msg);
    });

    picker.loadURL(`http://localhost:${PORT}/camera-picker.html`);

    // Listen for navigation to detect selection (more reliable than hash polling)
    picker.webContents.on('will-navigate', (e, url) => {
      e.preventDefault();
      const hash = new URL(url).hash;
      console.log('[picker] navigate hash:', hash);
      handleHash(hash);
    });

    // Also poll hash as backup
    const checkHash = () => {
      picker.webContents.executeJavaScript('window.location.hash').then((hash) => {
        if (hash && hash.length > 1) {
          console.log('[picker] hash poll:', hash);
          handleHash(hash);
        }
      }).catch(() => {});
    };

    const interval = setInterval(checkHash, 300);

    let resolved = false;
    function handleHash(hash) {
      if (resolved) return;
      if (hash.startsWith('#selected:')) {
        const payload = hash.slice('#selected:'.length);
        // Format: deviceId:encodedLabel
        const colonIdx = payload.indexOf(':');
        const deviceId = colonIdx >= 0 ? payload.slice(0, colonIdx) : payload;
        console.log('[picker] selected camera:', deviceId);
        resolved = true;
        clearInterval(interval);
        picker.close();
        resolve(deviceId || '');
      } else if (hash === '#skip') {
        console.log('[picker] skipped');
        resolved = true;
        clearInterval(interval);
        picker.close();
        resolve('');
      }
    }

    picker.on('closed', () => {
      clearInterval(interval);
      if (!resolved) {
        resolved = true;
        resolve('');
      }
    });
  });
}

function createMainWindow(cameraDeviceId) {
  console.log('[main] creating window, camera:', cameraDeviceId || '(none)');
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    title: 'DataOrb',
    icon: path.join(fluidRoot, 'build', 'icon.icns'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
    backgroundColor: '#000000',
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 12, y: 12 },
  });

  // Log renderer console to terminal
  mainWindow.webContents.on('console-message', (e, level, msg) => {
    console.log('[app]', msg);
  });

  const url = cameraDeviceId
    ? `http://localhost:${PORT}/#camera=${cameraDeviceId}`
    : `http://localhost:${PORT}/`;
  console.log('[main] loading URL:', url);

  // Retry loading if server isn't ready yet
  async function loadWithRetry(retries = 5) {
    for (let i = 0; i < retries; i++) {
      try {
        await mainWindow.loadURL(url);
        return;
      } catch (e) {
        console.log(`[main] loadURL attempt ${i + 1} failed:`, e.message);
        if (i < retries - 1) await new Promise(r => setTimeout(r, 1000));
      }
    }
  }
  loadWithRetry();

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  // Request camera access on macOS (with timeout to avoid hanging)
  if (process.platform === 'darwin') {
    const status = systemPreferences.getMediaAccessStatus('camera');
    console.log('[permissions] camera status:', status);
    if (status !== 'granted') {
      try {
        const granted = await Promise.race([
          systemPreferences.askForMediaAccess('camera'),
          new Promise(resolve => setTimeout(() => resolve('timeout'), 10000)),
        ]);
        console.log('[permissions] camera granted:', granted);
      } catch (e) {
        console.log('[permissions] camera request error:', e.message);
      }
    }
  }

  setupPermissions();

  // Start server
  const alreadyRunning = await isPortInUse();
  if (!alreadyRunning) {
    await startServer();
  } else {
    console.log('[server] already running on port', PORT);
  }

  // Show camera picker first
  const cameraDeviceId = await showCameraPicker();

  // Launch main app
  createMainWindow(cameraDeviceId);
});

app.on('window-all-closed', () => {
  if (serverProcess) serverProcess.kill();
  app.quit();
});

app.on('activate', () => {
  if (mainWindow === null) createMainWindow('');
});

app.on('before-quit', () => {
  if (serverProcess) serverProcess.kill();
});
