"use strict";

class App {

    // DOM Elements
    playerHolder;
    audio;
    loadingOverlay;

    // Widgets and controllers
    player;
    bookmarks;
    bookText;
    sleepTimer;
    zombieChecker;
    dragDropController;
    header;
    toast;
    overlayManager;
    menu;
    storageController;
    rootAttributer;
    playerVisibility;

    // State variables
    file = null;
    url = null;
    fileId = null;
    metaBookmarkIndices = [];
    isStarted = false;
    isLoading = false;
    pollIntervalId = -1;
    lastStorePositionTime = 0;

    constructor() {

        // DOM elements
        this.playerHolder = document.getElementById("playerHolder");
        this.audio = document.getElementById("audio");
        this.loadingOverlay = document.getElementById("loadingOverlay");

        // Controllers
        this.storageController = new StorageController();
        this.rootAttributer = new RootAttributer(document.documentElement, this.storageController)

        this.sleepTimer = new SleepTimer({
            audio: this.audio,
            rootAttributer: this.rootAttributer,
            sleepTimeLeft: document.getElementById("sleepTimeLeft"),
            onShowToast: (message) => this.toast.show(message),
            onUpdateMenu: () => this.menu.update()
        });
        this.zombieChecker = new ZombieChecker(this.audio, this.onZombieDetected);
        this.dragDropController = new DragDropController({
            overlay: document.getElementById('dragOverlay'),
            onDropFile: (file) => this.loadAudioFileOrUrl(file, null),
            isLoading: () => this.isLoading
        });

        // Widgets (view controllers for pre-existing DOM elements or groups of elements)
        this.player = new AudioPlayer({ audioElement: this.audio, container: this.playerHolder });
        this.bookmarks = new Bookmarks({
            mainEl: document.getElementById("bookmarkPanel"),
            onSelected: (index) => this.onBookmarkSelected(index),
            onChanged: () => this.onBookmarksChanged()
        });
        this.bookText = new BookText({
            textHolder: document.getElementById("textHolder"),
            fileNameLabel: document.getElementById("currentFileName"),
            onSeek: (time) => { this.audio.currentTime = time; },
            onPlay: () => this.playerPlay(),
            onStorePosition: (time) => this.storePosition(time),
            onShouldUpdateHighlight: () => !this.zombieChecker.isChecking,
            onAdvancedOnce: () => { this.overlayManager.collapseMenuButton(); },
            onIsPaused: () => this.audio.paused,
            onIsEnded: () => this.audio.ended,
            onPause: () => this.audio.pause()
        });
        this.header = new Header({
            mainEl: document.getElementById("loadHolder"),
            helpTemplate: document.getElementById("helpTemplate"),
            onUrlSubmit: (url) => this.loadAudioFileOrUrl(null, url),
            onFileChange: (file) => this.loadAudioFileOrUrl(file, null)
        });
        this.toast = new Toast({
            toast: document.getElementById("toast"),
            onPlay: () => this.playerPlay()
        });
        this.menu = new Menu({
            mainEl: document.getElementById("menuPanel"),
            rootAttributer: this.rootAttributer,
            storageController: this.storageController,
            onScrollTop: () => this.doScrollTop(),
            onHidePanels: () => this.overlayManager.hideAll(),
            onSleepTimerStart: () => this.sleepTimer.start(),
            onSleepTimerClear: () => this.sleepTimer.clear()
        });
        this.overlayManager = new OverlayManager({
            scrim: document.getElementById("scrim"),
            menu: this.menu,
            bookmarks: this.bookmarks,
            menuButton: document.getElementById("menuButton"),
            bookmarkButton: document.getElementById("bookmarkButton"),
            bookText: this.bookText,
            rootAttributer: this.rootAttributer,
            onOverlayStart: () => this.onOverlayStart(),
            onOverlayEnd: () => this.onOverlayEnd()
        });

        this.playerVisibility = new PlayerVisibilityController({
            playerOverlay: document.getElementById("playerOverlay"),
            playerHolder: this.playerHolder,
            player: this.player
        });

        // Listeners
        this.player.pinBtn.addEventListener("click", this.onPinButtonClick.bind(this));
        this.audio.addEventListener('play', this.onAudioPlay.bind(this));
        this.audio.addEventListener('pause', this.onAudioPause.bind(this));
        this.audio.addEventListener('error', this.onAudioError.bind(this));
        document.addEventListener("keydown", this.onKeyDown.bind(this));
        document.addEventListener('visibilitychange', this.onVisibilityChange.bind(this));

        // Page init
        if (Util.isTouchDevice()) {
            this.player.removePinButton();
        }

        const settings = this.storageController.loadDataAttributes();
        this.menu.applySettings(settings);

        const isPinned = this.storageController.loadPlayerPinned();
        this.player.setPinned(isPinned);

        this.reset();

        // Load audio from queryparam "url" if any
        const urlParams = new URLSearchParams(window.location.search);
        const url = urlParams.get("url");
        if (url) {
            this.loadAudioFileOrUrl(null, url);
        }
    }

    // ========================================

    /**
     * Resets page state to that of being freshly loaded
     */
    reset(dontAddHelp = false) {
    
        clearInterval(this.pollIntervalId);
        this.pollIntervalId = -1;
        
        this.audio.src = "";
        this.audio.load();

        this.isStarted = false;
        this.file = null;
        this.url = "";
        this.fileId = "";
        this.lastStorePositionTime = 0;

        this.rootAttributer.set("data-player-status", "none");

        if (document.activeElement && document.activeElement.blur) {
            document.activeElement.blur();
        }

        this.header.clearHelp();
        this.header.githubButton.style.display = "block"; 
        const lastFileName = this.storageController.loadLastOpened();
        this.header.showLastFileName(lastFileName);
        if (!dontAddHelp) {
            this.header.showHelpIfKeyboard();
        }

        this.bookText.hideFileName();
        this.bookText.clear();

        this.overlayManager.hideBookmarkButton();        
    }

    async loadAudioFileOrUrl(pFile, pUrl) {

        ShowUtil.hide(this.playerHolder, false);

        if (this.toast.isShowingPlayPrompt()) {
            this.toast.hide();
        }

        if (pUrl) {
            ShowUtil.show(this.loadingOverlay, "flex");
        }

        this.isLoading = true;
        const appMetadata = await MetadataUtil.loadAppMetadata(pFile, pUrl);
        this.isLoading = false;

        if (pUrl) {
            ShowUtil.hide(this.loadingOverlay);
        }

        if (!appMetadata || typeof appMetadata === 'string') {
            const errorMessage = appMetadata || "No tts-audiobook-tool metadata found";
            alert(errorMessage);
            this.syncAddressBar(null);
            return;
        }

        this.start(pFile, pUrl, appMetadata);
    }

    start(file, url, appMetadata) {
        /**
         * Initialize page with the already-loaded metadata, etc.
         */

        this.reset(true);

        this.file = file;
        this.url = url;

        this.fileId = Util.getObjectHash(appMetadata);
        this.storageController.setFileId(this.fileId);
        cl("fileId", this.fileId);
        
        const fileName = file ? file.name : url;
        this.storageController.saveLastOpened(fileName);

        this.syncAddressBar(url);

        this.header.hideLastFileName();
        this.header.githubButton.style.display = "none"; 

        // Init book text (heavy operation)
        const addSectionDividers = (appMetadata["has_section_break_audio"] === true);
        const textSegments = appMetadata["text_segments"];
        this.bookText.init(textSegments, addSectionDividers);
        this.bookText.showFileName(fileName)

        // Init bookmark-related
        this.metaBookmarkIndices = appMetadata["bookmarks"] || [];
        const shouldMeta = (!this.storageController.hasBookmarks() && this.metaBookmarkIndices.length > 0);
        let indices = null;
        if (shouldMeta) {
            indices = this.metaBookmarkIndices;
            this.storageController.saveBookmarks(indices);
            cl("seeded bookmarks from metadata", indices);
        } else {
            indices = this.storageController.loadBookmarks();
        }
        this.bookmarks.init(textSegments, indices);
        this.overlayManager.showBookmarkButton();
        this.bookText.addBookmarkClasses(this.bookmarks.indices);

        if (Util.isTouchDevice() || this.player.isPinned()) {
            this.playerVisibility.show();
        }

        this.audio.src = url || URL.createObjectURL(file);
        this.playerPlay();

        const time = this.storageController.loadPosition();
        if (time) {
            this.audio.currentTime = time;
        }

        clearInterval(this.pollIntervalId);
        this.pollIntervalId = setInterval(this.poll.bind(this), 50);

        this.isStarted = true;
    }

    // ========================================
    // Event Handlers
    // ========================================

    onKeyDown(event) {
        if (event.target.tagName == "INPUT") {
            return;
        }

        if (event.key == "Escape" && this.overlayManager.isScrimShowing()) {
            this.overlayManager.hideAll();
            return;
        }

        const doLoadLocal = function() {
            event.preventDefault();
            this.header.doFileButtonClick()
        }.bind(this);

        if (document.activeElement == this.header.loadLocalButtonLabel) {
            if (event.key === "Enter" || event.key === " ") {
                doLoadLocal();
                return;
            }
        }
        if (event.key == "o") {
            doLoadLocal();
            return;
        }

        if (!this.isStarted) {
            return;
        }

        switch (event.key) {
            case "Escape":
                if (this.audio.paused) {
                    this.playerPlay();
                } else {
                    this.audio.pause();
                }
                event.preventDefault();
                break;
            case ",":
                this.bookText.seekPreviousSegment();
                break;
            case ".":
                this.bookText.seekNextSegment();
                break;
            case "[":
                this.audio.currentTime -= 60;
                break;
            case "]":
                this.audio.currentTime += 60;
                break;
            case "b":
                this.overlayManager.toggleBookmarks();
        }
    }

    onAudioPlay() {
        if (this.zombieChecker.isChecking) {
            return;
        }
        if (this.toast.isShowingPlayPrompt()) {
            this.toast.hide();
        }
        this.rootAttributer.set("data-player-status", "play");
    }

    onAudioPause() {
        if (this.zombieChecker.isChecking) {
            return;
        }
        this.rootAttributer.set("data-player-status", "pause");
    }

    onAudioError(e) {
        // error = e.target.error
    }

    onVisibilityChange() {
        if (document.visibilityState === 'visible') {
            if (Util.isTouchDevice()) { 
                this.zombieChecker.check();
            }
        }
    }

    onPinButtonClick() {
        const b = !this.player.isPinned();
        this.player.setPinned(b);
        this.storageController.savePlayerPinned(b);
        if (!b) {
            this.playerVisibility.hide();
        }
    }

    onBookmarkSelected(index) {
        this.bookText.seekBySegmentIndex(index);
        this.audio.play();
        this.overlayManager.hideAll();
    }

    onBookmarksChanged(e) {
        if (this.bookmarks.indices.length == 0 && this.metaBookmarkIndices.length > 0) {
            if (confirm("Re-add file's embedded bookmarks?")) {
                this.bookmarks.initIndices(this.metaBookmarkIndices);
            }
        }
        this.storageController.saveBookmarks(this.bookmarks.indices);
    }

    onZombieDetected = (originalTime) => {
        if (this.file) {
            let s = "The browser has dropped the handle to the local audio file.\n\n";
            s += "This can occur when the mobile browser tab is put into the background while the audio is paused, unfortunately.\n\n";
            s += "Please load the file again to resume.";
            alert(s);
            window.location.reload();
        } else {
            // "Reload" the audio src to resolve
            this.audio.src = this.url;
            this.audio.load();
            this.audio.currentTime = originalTime;
        }
    }

    // ========================================

    poll() {
        if (this.zombieChecker.isChecking) {
            return;
        }

        this.playerVisibility.updateVisibility();

        if (!this.audio.ended && new Date().getTime() - this.lastStorePositionTime > 5000) {
            this.storePosition();
        }

        // Update highlighted text segment
        this.bookText.updateHighlight(this.audio.currentTime);
    }

    syncAddressBar(audioFileUrl) {
        const url = new URL(window.location.href);
        url.search = '';
        url.hash = '';

        if (audioFileUrl) {
            url.searchParams.set('url', audioFileUrl);
        }
        window.history.replaceState(null, '', url.toString());
    }

    doScrollTop() {
        window.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
    }

    storePosition(value) {
        value = value || this.audio.currentTime;
        this.storageController.storePosition(value);
        this.lastStorePositionTime = new Date().getTime();
    }

    async playerPlay() {
        try {
            await this.audio.play();
        } catch (error) {
            if (error.name == "NotAllowedError") {
                this.toast.show("Click to play audio", true);
            } else if (error.name == "NotSupportedError") {
                const s = `Error: ${error.name}\nCode: ${error.code}\n\nMessage: ${error.message}`;
                alert(s);
                this.storageController.clearLastOpened();
                this.reset();
            } else {
                console.error("audio.play() error - code:", error.code, "name:", error.name, "message:", error.message);
            }
        }
    }

    onOverlayStart() {
        document.body.classList.add("bodyNoScroll");
        document.getElementById('loadLocalButtonLabel').setAttribute("inert", "");
        this.player.makeButtonsInert(true);
    }

    onOverlayEnd() {
        document.body.classList.remove("bodyNoScroll");
        document.getElementById('loadLocalButtonLabel').removeAttribute("inert");
        this.player.makeButtonsInert(false);
    }
}

function cl(...rest) {
    console.log(...rest); 
}
