"use strict";

window.app = function() {

    const SLEEP_MS = (1000 * 60) * 15

    const root = document.documentElement;
    const loadFileInput = document.getElementById("loadFileInput");
    const lastFileNameHolder = document.getElementById("lastFileNameHolder")
    const sleepTimeLeft = document.getElementById("sleepTimeLeft");
    const currentFileNameDiv = document.getElementById("currentFileName")
    const playerOverlay = document.getElementById("playerOverlay"); // always visible
    const playerHolder = document.getElementById("playerHolder"); // animates in and out (translateY)
    const audio = document.getElementById("audio");
    const textHolder = document.getElementById("textHolder");
    const loadLocalButtonLabel = document.getElementById("loadLocalButtonLabel");
    const loadUrlInput = document.getElementById("loadUrlInput");
    const helpHolder = document.getElementById("helpHolder");
    const helpTemplate = document.getElementById("helpTemplate");
    const githubButton = document.getElementById("githubCorner");

    // Overlay-related UI elements
    const menuButton = document.getElementById("menuButton");
    const scrim = document.getElementById("scrim");
    const menuPanel = document.getElementById("menuPanel");
    const scrollTopButton = document.getElementById("scrollTopButton")
    const textSizeButton = document.getElementById("textSizeButton")
    const themeButton = document.getElementById("themeButton");
    const segmentColorsButton = document.getElementById("segmentColorsButton")
    const sleepButton = document.getElementById("sleepButton");
    const bookmarkPanel = document.getElementById("bookmarkPanel");
    const bookmarkButton = document.getElementById("bookmarkButton");
    const toast = document.getElementById("toast");
    const loadingOverlay = document.getElementById("loadingOverlay")

    // eslint-disable-next-line
    const player = new AudioPlayer({ audioElement: audio, container: playerHolder });

    // eslint-disable-next-line
    const bookmarkController = new BookmarkController(bookmarkPanel);

    let file = null;
    let url = null;
    let fileId = null; // id used to track the loaded resource
    let legacyFileId = null; // back compat-related

    let textSegments = [];
    let spans = []; // cached text segment span refs
    let metadataBookmarks = []; // bookmarks from the metadata (ie, not manually added by user)

    let isStarted = false; // is audiobook populated and audio loaded
    let isLoading = false;
    let hasAdvancedOnce = false; // has audio playback advanced one segment at least once
    let isToastPlayPrompt = false;

    let currentIndex = -1; // segment index whose time range encloses the player's currentTime
    let directSelections = []

    let toastHideDelayId = -1
    let sleepIntervalId = -1;
    let sleepEndTime = -1;
    let pollIntervalId = -1;
    let isInPlayer = false;
    let isCheckingZombie = false;
    let lastStorePositionTime = 0
    const mousePosition = { x: -1, y: -1};
    let dragCounter = 0;
    let isIgnoreDrag = false;

    // ********
    pageInit();
    // ********

    function pageInit() {

        initListeners();

        if (isTouchDevice()) {
            player.removePinButton();
        }

        const keys = Object.keys(localStorage);
        keys.forEach(key => {
            const value = localStorage[key]
            if (key.startsWith("data-")) {
                root.setAttribute(key, value)
            }
        });

        const s = localStorage.getItem("pref_player_pinned");
        const isPinned = (s === "1") || !s; // default is pinned
        player.setPinned(isPinned);

        initClickListeners();

        reset();

        // Load audio from queryparam "url" if any
        const urlParams = new URLSearchParams(window.location.search);
        const url = urlParams.get("url");
        if (url) {
            loadAudioFileOrUrl(null, url);
        }
    }

    function initListeners() {
        window.addEventListener('mousemove', (event) => {
            mousePosition.x = event.clientX;
            mousePosition.y = event.clientY;
        });

        loadUrlInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                const url = loadUrlInput.value.trim()
                if (url) {
                    loadUrlInput.blur();
                    loadAudioFileOrUrl(null, url);
                }
            }
        });

        loadUrlInput.addEventListener('blur', () => {
            loadUrlInput.value = '';
        });

        loadFileInput.addEventListener('change', async () => {
            const file = loadFileInput.files[0];
            if (file) {
                // The user has selected a local file
                loadAudioFileOrUrl(file, null);
            }
        });

        // Force <label> to un-focus post-file-requestor, ffs
        loadFileInput.addEventListener('click', () => {
            const onFocusBack = () => {
                window.removeEventListener('focus', onFocusBack);
                setTimeout(() => {
                    loadLocalButtonLabel.blur()
                }, 1)
              };
              window.addEventListener('focus', onFocusBack);
        })

        audio.addEventListener('play', function() {
            if (isCheckingZombie) {
                return;
            }
            if (isToastPlayPrompt) {
                isToastPlayPrompt = false;
                hideToast();
            }
            root.setAttribute("data-player-status", "play");
            currentIndex = -1; // ensures scroll to current segment
        });

        audio.addEventListener('pause', function() {
            if (isCheckingZombie) {
                return;
            }
            root.setAttribute("data-player-status", "pause");
        });

        audio.addEventListener('error', function(e) {
            // error = e.target.error
            // console.log("Audio error:", error.code, error.message)
        });

        document.addEventListener("bookmarkSelect", (e) => { onBookmarkSelect(e); });
        document.addEventListener("bookmarksChanged", (e) => { onBookmarksChanged(e); });

        textHolder.addEventListener('click', onTextClick);

        document.addEventListener("keydown", onKeyDown);

        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') {
                // Browser tab has gone from not-visible to visible
                // Note, checking zombie only when is-touch-device (ie, mobile, ie, aggressively kills file handle)
                const should = (!isCheckingZombie && isTouchDevice() && audio.duration > 0 && audio.paused && !audio.ended);
                if (should) {
                    checkZombieState();
                }
            }
        });

        initDragDropListeners();
    }

    function initClickListeners() {

        player.pinBtn.addEventListener("click", (e) => {
            const b = !player.isPinned();
            player.setPinned(b);
            localStorage.setItem("pref_player_pinned", b ? "1" : "0");
            if (!b) {
                hidePlayer();
            }
        });

        scrim.addEventListener("click", (e) => {
            e.stopPropagation();
            hideScrimAndPanels();
        });

        // eat. // TODO: is this still necessary?
        menuPanel.addEventListener("click", (e) => { e.stopPropagation(); });
        bookmarkPanel.addEventListener("click", (e) => { e.stopPropagation() });

        menuButton.addEventListener('click', (e) => {
            collapseOptionsButton();
            if (!isElementShowing(menuPanel)) {
                showMenuPanel();
            } else {
                hideScrimAndPanels();
            }
        });

        bookmarkButton.addEventListener('click', (e) => { toggleBookmarkPanel(); });

        // Menu buttons
        scrollTopButton.addEventListener('click', (e) => {
            e.stopPropagation();
            hideScrimAndPanels();
            window.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
        });
        textSizeButton.addEventListener('click', (e) => {
            e.stopPropagation();
            cycleRootAttribute("data-book-text-size", ["medium", "small"]);
            updateMenuButtons()
        });
        themeButton.addEventListener('click', (e) => {
            e.stopPropagation();
            cycleRootAttribute("data-theme", ["dark"])
            updateMenuButtons()
        });
        segmentColorsButton.addEventListener('click', (e) => {
            e.stopPropagation();
            cycleRootAttribute("data-segment-colors", ["blue", "red"])
            updateMenuButtons()
        });
        sleepButton.addEventListener("click", (e) => {
            e.stopPropagation();
            const value = cycleRootAttribute("data-sleep", ["on"], false);
            if (value == "on") {
                startSleep();
            } else {
                clearSleep()
            }
            updateMenuButtons();
            hideScrimAndPanels();
        });

        const loadUrlIcon = document.getElementById("loadUrlIcon");
        if (loadUrlIcon) {
            loadUrlIcon.addEventListener("click", () => { showLoadUrlInfo() })
        }

        toast.addEventListener("click", () => {
            if (isToastPlayPrompt) {
                audio.play();
            }
            hideElement(toast);
        });
    }

    function initDragDropListeners() {
        
        const overlay = document.getElementById('dragOverlay');

        window.addEventListener('dragenter', (e) => {
            e.preventDefault();
            isIgnoreDrag = (isLoading || isCheckingZombie);
            dragCounter++;
            if (isIgnoreDrag) {
                return;
            }
            overlay.style.display = 'block';
        });
        window.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dragCounter--;
            if (dragCounter === 0) {
                overlay.style.display = 'none';
                if (isIgnoreDrag) {
                    isIgnoreDrag = false;
                }
            }
        });
        window.addEventListener('dragover', (e) => {
            e.preventDefault(); // required
        });

        window.addEventListener('drop', (e) => {
            e.preventDefault();
            
            // Reset drag state
            dragCounter = 0;
            overlay.style.display = 'none';

            if (isIgnoreDrag) {
                isIgnoreDrag = false;
                return;
            }

            const file = e.dataTransfer.files[0];
            loadAudioFileOrUrl(file, null);
        });
    }

    /**
     * Reset page state; should be tantamount to a 'reload', mostly
     */
    function reset(dontAddHelp=false) {

        clearInterval(pollIntervalId);
        pollIntervalId = -1;
        isStarted = false;

        clearInterval(sleepIntervalId);
        sleepIntervalId = -1;
        sleepEndTime = -1;

        // Clear audio
        // src="" + load() aborts any pending activity
        // Triggers browser console warning, can't be helped
        audio.src = "";
        audio.load();

        file = null;
        url = "";
        fileId = "";
        legacyFileId = "";
        currentIndex = -1;
        textSegments = [];
        directSelections = [];
        isToastPlayPrompt = false;
        lastStorePositionTime = 0
        isCheckingZombie = false;

        root.setAttribute("data-player-status", "none");
        githubButton.style.display = "block";
        currentFileNameDiv.style.display = "none"
        clearText();
        hideElement(bookmarkButton);
        if (document.activeElement && document.activeElement.blur) { // remove any focus
            document.activeElement.blur();
        }

        // Update 'last opened' text
        const s = localStorage.getItem("last_opened") 
        if (s) {
            document.getElementById("lastFileNameText").textContent = s;
            lastFileNameHolder.style.display = "block";
        }
        else {
            document.getElementById("lastFileNameText").textContent = s;
            lastFileNameHolder.style.display = "none";
        }

        // Add help info
        if (!dontAddHelp) {
            addHelpIfKeyboard();
        }
    }

    /**
     * pFile and pUrl are mutually exclusive
     */
    async function loadAudioFileOrUrl(pFile, pUrl) {

        hidePlayer();

        if (isToastPlayPrompt) {
            isToastPlayPrompt = false;
            hideToast();
        }

        if (pUrl) {
            showElement(loadingOverlay, "flex");
        }

        isLoading = true;
        // eslint-disable-next-line
        const appMetadata = await loadAppMetadata(pFile, pUrl);
        isLoading = false;

        if (pUrl) {
            hideElement(loadingOverlay);
        }

        if (!appMetadata || typeof appMetadata === 'string') {
            const errorMessage = appMetadata || "No tts-audiobook-tool metadata found";
            alert(errorMessage);
            return;
        }

        start(pFile, pUrl, appMetadata);
    }

    /**
     * pFile and pUrl are mutually exclusive
     */
    function start(pFile, pUrl, appMetadata) {

        // Reset page state
        reset(true);

        file = pFile;
        url = pUrl;
        fileId = getObjectHashSync(appMetadata);
        legacyFileId = file ? file.name : url;
        console.log("fileId", fileId)

        textSegments = appMetadata["text_segments"];

        localStorage.setItem("last_opened", file ? file.name : url);

        helpHolder.innerHTML = "";
        githubButton.style.display = "none";
        lastFileNameHolder.style.display = "none";

        currentFileNameDiv.style.display = "block"
        currentFileNameDiv.textContent = file ? file.name : url

        const addSectionDividers = (appMetadata["has_section_break_audio"] === true);
        populateText(addSectionDividers);

        // Bookmarks
        const noBookmarksYet = (localStorage.getItem(`bookmarks_fileId_${fileId}`) === null);
        metadataBookmarks = appMetadata["bookmarks"] || [];        
        const shouldPopulateMetadataBookmarks = (noBookmarksYet && metadataBookmarks.length > 0);
        let bookmarks = null;
        if (shouldPopulateMetadataBookmarks) {
            bookmarks = metadataBookmarks
            saveBookmarks()
            console.log("inited bookmarks", bookmarks)
        } else {
            bookmarks = loadBookmarks()
        }
        bookmarkController.init(textSegments, bookmarks);
        showElement(bookmarkButton);
        addBookmarkClasses();

        if (isTouchDevice() || player.isPinned()) {
            showPlayer();
        }

        // Play
        audio.src = url || URL.createObjectURL(file)
        playerPlay();

        // Seek to stored position if exists
        let time = localStorage.getItem("fileId_" + fileId)
        if (time) {
            time = parseFloat(time);
            if (time) {
                audio.currentTime = time;
            }
        }

        clearInterval(pollIntervalId);
        pollIntervalId = setInterval(poll, 50);

        isStarted = true;
    }

    function populateText(addSectionDividers) {

        let contentHtml = '';
        textSegments.forEach((segment, i) => {

            const o = splitWhitespace(segment.text)

            if (o["before"]) {
                contentHtml += escapeHtml( o["before"] );
            }

            const hasAudio = segment["time_start"] > 0.0 && segment["time_end"] > 0.0
            const className = hasAudio ? "hasAudio" : "noAudio"
            const spanString = `<span id="segment-${i}" class="${className}">${ escapeHtml( o["content"] ) }</span>`
            contentHtml += spanString;

            if (o["after"]) {

                if (addSectionDividers) {
                    const numLfs = o["after"].split('\n').length - 1;
                    if (numLfs >= 3) {
                        // Two+ blank lines - treat as 'section break'
                        // TODO: add the "segment reason" field to the metadata rather than inferring this
                        contentHtml += "<br>&nbsp;<hr><br>";
                    } else {
                        contentHtml += escapeHtml( o["after"] );
                    }
                    } else {
                        contentHtml += escapeHtml( o["after"] );
                }
            }
        });

        textHolder.innerHTML = contentHtml;
        textHolder.style.display = "block";

        // Cache span references
        spans = [];
        for (let i = 0; i < textSegments.length; i++) {
            spans[i] = document.getElementById("segment-" + i);
        }
    }

    function clearText() {
        textHolder.innerHTML = "";
        textHolder.style.display = "none";
        spans = []
    }

    function updateMenuButtons() {
        setButtonChildVisible(textSizeButton, root.getAttribute("data-book-text-size"));
        setButtonChildVisible(themeButton, root.getAttribute("data-theme"));
        setButtonChildVisible(sleepButton, root.getAttribute("data-sleep"));
        // rem, segment colors button does not multiple children
    }

    // --------------------------------------

    function showScrim() {
        document.body.classList.add("bodyNoScroll");
        loadLocalButtonLabel.setAttribute("inert", "");
        player.makeButtonsInert(true);
        showElement(scrim);
    }

    function hideScrimAndPanels() {
        // TODO: re-enable them
        document.body.classList.remove("bodyNoScroll");
        loadLocalButtonLabel.removeAttribute("inert");
        player.makeButtonsInert(false);
        
        hideElement(scrim);
        hideElement(menuPanel);
        
        hideElement(bookmarkPanel);
        addBookmarkClasses();
    }

    function showMenuPanel() {
        updateMenuButtons();
        document.body.classList.add("bodyNoScroll"); // Prevent background scrolling
        showScrim();
        hideElement(bookmarkPanel);
        showElement(menuPanel, "flex");
    }

    function toggleBookmarkPanel() {
        if (!isElementShowing(bookmarkPanel)) {
            showBookmarkPanel();
        } else {
            hideScrimAndPanels();
        }
    }

    function showBookmarkPanel() {
        document.body.classList.add("bodyNoScroll");
        removeBookmarkClasses();
        bookmarkController.updateAddButton(currentIndex);
        showScrim();
        hideElement(menuPanel);
        showElement(bookmarkPanel)
    }

    /**
     * Shows toast and hides it after a delay
     */
    function showToast(message, isPlayPrompt=false) {
        clearTimeout(toastHideDelayId);
        isToastPlayPrompt = isPlayPrompt;
        toast.textContent = message;

        showElement(toast);
        if (!isPlayPrompt) {
            toastHideDelayId = setTimeout(() => { hideToast(); }, 2500);
        }
    }

    function hideToast() {
        clearTimeout(toastHideDelayId);
        isToastPlayPrompt = false;
        hideElement(toast);
    }

    function showLoadUrlInfo() {
        let s= "For easy sharing...\n\n"
        s += "Append ?url= followed by your hosted file's URL to share your audiobook with others.\n\n"
        s += "CORS info:\n\n";
        s += `When hosting audio files to be used by the player, remember to set "Access-Control-Allow-Origin" appropriately.\n\n`;
        s += `For example:\n`;
        s += `Access-Control-Allow-Origin: ${CORS_GITHUB_URL}, http://192.168.1.2:8000\n`;
        s += `or: \n`;
        s += `Access-Control-Allow-Origin: *`;
        alert(s);
    }

    // --------------------------------------

    function onKeyDown(event) {

        if (event.target.tagName == "INPUT") {
            return;
        }

        if (event.key == "Escape" && isElementShowing(scrim)) {
            hideScrimAndPanels();
            return;
        }

        const doLoadLocal = function() {
            event.preventDefault();
            loadFileInput.click();
            loadLocalButtonLabel.blur();
        };
        if (document.activeElement == loadLocalButtonLabel) {
            if (event.key === "Enter" || event.key === " ") {
                doLoadLocal()
                return;
            }
        }
        if (event.key == "o") {
            doLoadLocal()
            return;
        }

        if (!isStarted) {
            return;
        }

        // Hotkeys that are active only while audio is loaded
        switch (event.key) {
            case "Escape":
                if (audio.paused) {
                    playerPlay();
                } else {
                    audio.pause();
                }
                event.preventDefault();
                break;
            case ",":
                seekPreviousSegment();
                break;
            case ".":
                seekNextSegment();
                break;
            case "`":
                if (directSelections.length > 0) {
                    const index = directSelections.pop();
                    seekBySegmentIndex(index);
                }
                break;
            case "[":
                audio.currentTime -= 60;
                break;
            case "]":
                audio.currentTime += 60;
                break;
            case "b":
                toggleBookmarkPanel();
        }
    }

    function onTextClick(event) {

        const clickedSpan = event.target;

        const isSegment = (clickedSpan.tagName === 'SPAN' && clickedSpan.id.startsWith("segment-"));
        if (!isSegment) {
            return;
        }

        const segmentIndex = parseInt(clickedSpan.id.split("-")[1]);
        const segment = textSegments[segmentIndex];

        if (segment["time_start"] == 0 && segment["time_end"] == 0) {
            return;
        }

        if (clickedSpan == getCurrentSpan()) {
            // Toggle play
            if (audio.paused) {
                playerPlay();
            } else {
                audio.pause()
            }
        } else {
            seekBySegmentIndex(segmentIndex, true);

            const isSameAsLast = (directSelections.at(-1) !== undefined) && (directSelections.at(-1) == segmentIndex)
            if (!isSameAsLast) {
                directSelections.push(segmentIndex);
            }
        }
    }

    function onBookmarkSelect(e) {
        const s = e.detail.index;
        const index = parseInt(s);
        seekBySegmentIndex(index);
        audio.play();
        hideScrimAndPanels();
    }

    function onBookmarksChanged(e) {
        if (bookmarkController.indices.length == 0 && metadataBookmarks.length > 0) {
            if (confirm("Re-add file's embedded bookmarks?")) {
                bookmarkController.initIndices(metadataBookmarks);
            }
        }
        saveBookmarks();        
    }

    function poll() {
        // Checks if audio position has crossed text segment boundaries
        // Also checks for mouse-over-playerOverlay

        if (isCheckingZombie) {
            return;
        }

        updatePlayerVisibility();

        // Store audio position at least every 5 seconds
        if (!audio.ended && new Date().getTime() - lastStorePositionTime > 5000) {
            storePosition();
        }

        const previousIndex = currentIndex
        currentIndex = getSegmentIndexBySeconds(audio.currentTime);
        if (currentIndex == previousIndex) {
            return;
        }

        // Index has changed:

        unhighlightByIndex(previousIndex);

        if (currentIndex - previousIndex == 1) {
            // Has advanced by one text segment, presumably due to normal playback
            if (!hasAdvancedOnce) {
                hasAdvancedOnce = true;
                collapseOptionsButton();
            }
            if (!audio.ended) {
                storePosition();
            }
        }

        // Highlight active and scroll-to
        if (currentIndex >= 0) {
            getCurrentSpan()?.classList.add("highlight");
            if (!(document.activeElement instanceof HTMLInputElement)) {
                getCurrentSpan().scrollIntoView({
                    behavior: 'smooth',
                    block: 'center', // 'start', 'center', 'end', or 'nearest'
                    inline: 'nearest' // 'start', 'center', 'end', or 'nearest'
                });
            }
        }
    }

    // -------------------------------------

    function storePosition(value) {
        value = value || audio.currentTime;
        localStorage.setItem("fileId_" + fileId, value);
        lastStorePositionTime = new Date().getTime();
    }

    function collapseOptionsButton() {
        root.setAttribute("data-ui-panel-button-collapse", "true");
    }

    function getSpanByIndex(i) {
        return i >= 0 ? spans[i] : null;
    }

    function getCurrentSpan() {
        return currentIndex >= 0 ? spans[currentIndex] : null;
    }

    function getSegmentIndexBySeconds(seconds) {
        // Starts search from current index and fans out (optimization)

        if (textSegments.length == 0) {
            return -1;
        }

        const baseIndex = Math.max(currentIndex, 0);

        let delta = 0;
        while (true) {

            const indexInc = baseIndex + delta
            const indexDec = baseIndex - delta;

            const oob = (indexInc >= textSegments.length && indexDec < 0)
            if (oob) {
                return -1;
            }

            if (indexDec >= 0) {
                const segment = textSegments[indexDec];
                if (seconds >= segment["time_start"] && seconds < segment["time_end"]) {
                    return indexDec
                }
            }
            if (indexInc < textSegments.length) {
                const segment = textSegments[indexInc];
                if (seconds >= segment["time_start"] && seconds < segment["time_end"]) {
                    return indexInc
                }
            }

            delta += 1;
        }
    }

    /**
     * Seeks to the next segment that has a starting time
     */
    function seekNextSegment() {
        let index = currentIndex; // rem, index can be -1
        for (let i = index + 1; i <= index + 100; i++) {
            if (i >= textSegments.length) {
                return;
            }
            const segment = textSegments[i]
            const has_time = (segment["time_end"] > 0);
            if (has_time) {
                seekBySegmentIndex(i);
                return;
            }
        }
    }

    function seekPreviousSegment() {
        if (currentIndex <= 0) {
            return
        }
        for (let i = currentIndex - 1; i >= currentIndex - 100; i--) {
            if (i < 0) {
                return;
            }
            const segment = textSegments[i]
            const has_time = (segment["time_end"] > 0);
            if (has_time) {
                seekBySegmentIndex(i);
                return;
            }
        }
    }

    /**
     * Unhighlights specified segment
     * plus its close neighbors, for reasons.
     */
    function unhighlightByIndex(i) {
        const a = Math.max(i - 20, 0)
        const b = Math.min(i + 20, textSegments.length - 1)
        for (let i = a; i <= b; i++) {
            spans[i].classList.remove("highlight");
        }
    }

    function seekBySegmentIndex(i, andPlay) {
        unhighlightByIndex(currentIndex);
        const targetTime = textSegments[i]["time_start"];
        audio.currentTime = targetTime;
        if (audio.paused && andPlay) {
            playerPlay();
        }
    }

    async function playerPlay() {
        try {
            await audio.play();
        } catch (error) {
            if (error.name == "NotAllowedError") {
                showToast("Click to play audio", true);
            } else if (error.name == "NotSupportedError") {
                const s = `Error: ${error.name}\nCode: ${error.code}\n\nMessage: ${error.message}`;
                alert(s)
                localStorage.setItem("last_opened", ""); // Roll back value
                reset();
            } else {
                console.error("audio.play() error - code:", error.code, "name:", error.name, "message:", error.message)
            }
        }
    }

    function startSleep() {
        showToast("Sleep timer: Will auto-pause in 15 minutes");

        sleepEndTime = new Date().getTime() + SLEEP_MS;

        clearInterval(sleepIntervalId);
        onSleepInterval();
        sleepIntervalId = setInterval(onSleepInterval, 1000);
    }

    function clearSleep(isFinishedMessage=false) {
        const message = isFinishedMessage ? "Sleep timer finished" : "Sleep timer cancelled";
        showToast(message);

        root.setAttribute("data-sleep", "");
        clearInterval(sleepIntervalId);
        updateMenuButtons();
    }

    function onSleepInterval() {
        const ms = sleepEndTime - new Date().getTime();
        if (ms <= 0) {
            audio.pause();
            clearSleep(true);
            return;
        }
        sleepTimeLeft.textContent = msToString(ms);
    }

    function loadBookmarks() {
        if (!fileId) {
            return [];
        }
        const key = `bookmarks_fileId_${fileId}`;
        const value = localStorage.getItem(key);
        if (!value) {
            return [];
        }
        let arr = [];
        try {
            arr = JSON.parse(value);
        } catch (exc) {
            console.log("json exception:", value)
            return [];
        }
        if (!Array.isArray(arr)) {
            console.log("not an array:", arr)
            return [];
        }
        return arr;
    }

    function saveBookmarks() {
        if (!fileId) {
            return;
        }
        const key = `bookmarks_fileId_${fileId}`;
        const value = JSON.stringify(bookmarkController.indices);
        localStorage.setItem(key, value);
    }

    function addBookmarkClasses() {
        for (const index of bookmarkController.indices) {
            spans[index].classList.add("bookmark");
        }
    }

    function removeBookmarkClasses() {
        for (const index of bookmarkController.indices) {
            spans[index].classList.remove("bookmark");
        }
    }

    // ----------------------------------------

    /**
     * Cycles between [none], values[0], values[1], etc.
     * Returns the value which was set on the attribute
     */
    function cycleRootAttribute(attribute, values, andPersist=true) {

        if (!Array.isArray(values) || values.length == 0) {
            console.warning("bad value for `values`");
            return;
        }

        const currentValue = root.getAttribute(attribute);
        let currentIndex = values.indexOf(currentValue);

        let nextIndex;
        if (currentIndex == -1) {
            nextIndex = 0;
        } else {
            nextIndex = currentIndex + 1;
            if (nextIndex >= values.length) {
                nextIndex = -1;
            }
        }

        let targetValue;
        if (nextIndex == -1) {
            targetValue = ""
        } else {
            targetValue = values[nextIndex];
        }

        // console.log('currentvalue', currentValue)
        // console.log('currentindex', currentIndex)
        // console.log('nextindex', nextIndex)
        // console.log('targetvalue', targetValue)

        if (targetValue == "") {
            root.removeAttribute(attribute);
        } else {
            root.setAttribute(attribute, targetValue);
        }

        if (andPersist) {
            localStorage.setItem(attribute, targetValue);
        }

        return targetValue;
    }

    /**
     * Returns the value which was set, if any
     */
    function setRootAttributeFromLocalStorage(dataAttribute) {
        const value = localStorage.getItem(dataAttribute)
        if (value) {
            root.setAttribute(dataAttribute, value)
            return value;
        }
        return null;
    }

    /**
     *
     */
    function setButtonChildVisible(holder, targetValue) {

        for (const child of holder.children) {
            child.style.display = "none";
        }

        for (const child of holder.children) {

            const value = child.dataset["value"]

            let isMatch = (value === targetValue);
            isMatch |= (!value || value == "default") && (!targetValue || targetValue == "default");
            if (isMatch) {
                child.style.display = 'revert';
                break;
            }
        }
    }

    /**
     * Checks if <audio> is in a 'zombie state' (ie, if it no longer has a 'handle' on the data)
     * Only way to verify is to observe its behavior after resuming playback.
     */
    function checkZombieState() {

        if (audio.duration == 0) {
            return
        }

        isCheckingZombie = true;
        let timeoutId = 0;
        const originalCurrentTime = audio.currentTime;
        const originalVolume = audio.volume;
        let timeUpdateCount = 0;

        const cleanUp = () => {
            audio.removeEventListener("timeupdate", onTimeUpdate);
            audio.removeEventListener("ended", onEnded);
            clearTimeout(timeoutId);

            isCheckingZombie = false;
            audio.pause();
            audio.volume = originalVolume;
        };

        const doFailed = () => {

            cleanUp();

            if (file) {
                // Unrecoverable because reloading local file requires user intervention
                let s = "The browser has dropped the handle to the local audio file.\n\n";
                s += "This can occur when the mobile browser tab is put into the background while the audio is paused, unfortunately.\n\n";
                s += "Please load the file again to resume.";
                alert(s);
                window.location.reload();

            } else {
                // Is url
                // TODO: untested (have not been able to replicate but have gotten reports that it does happen on iOS)
                audio.url = url;
                audio.load(); // for good measure
                audio.currentTime = originalCurrentTime;
            }
        };

        const onEnded = () => {
            doFailed();
        }

        const onTimeUpdate = () => {
            if (audio.currentTime >= audio.duration) {
                doFailed();
                return;
            }
            timeUpdateCount += 1;
            if (timeUpdateCount >= 3) {
                // Calling it good
                // (It's possible only one occurrence is necessary, but yea)
                cleanUp();
            }
        };

        const onTimeout = () => {
            // Failsafe. It's also possible browser needed more time to resume, but let's not
            cleanUp();
        };

        audio.addEventListener("timeupdate", onTimeUpdate);
        audio.addEventListener("ended", onEnded, { once: true } );
        timeoutId = setTimeout(onTimeout, 1000);
        audio.volume = 0.0;
        audio.play(); // Minor FYI: Will fail if source is url and has not yet been user-activated
    }

    // --------------------------------------
    // Player show/hide logic

    function updatePlayerVisibility() {

        if (isTouchDevice() ||  player.isPinned()) {
            return;
        }

        const wasInPlayer = isInPlayer;
        isInPlayer = isMouseOverElement(playerOverlay); // TODO && focus
        if (isInPlayer == wasInPlayer) {
            return
        }

        if (isInPlayer) {
            showPlayer();
        } else {
            hidePlayer();
        }
    }

    function showPlayer() {
        showElement(playerHolder);
    }

    function hidePlayer() {
        hideElement(playerHolder, false);
    }

    function addHelpIfKeyboard() {
        helpHolder.innerHTML = "";
        if (hasPersistentKeyboard()) {
            const help = helpTemplate.content.cloneNode(true);
            helpHolder.appendChild(help);
        }
    }

    // --------------------------------------
    // 'Utility' functions
    // TODO: move to dedicated script after implementing script concat

    function isTouchDevice() {
        // hand-wavey test
        return !matchMedia("(pointer:fine)").matches
    }

    function hasPersistentKeyboard() {
        // hand-wavey test
        const hasFinePointer = window.matchMedia("(pointer: fine)").matches;
        const hasHoverSupport = window.matchMedia("(hover: hover)").matches;
        return (hasFinePointer && hasHoverSupport);
    }

    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#39;")
    }

    function splitWhitespace(str) {
        // Match leading whitespace, content, and trailing whitespace
        str = str || "";
        const match = str.match(/^(\s*)(.*?)(\s*)$/);
        if (!match) {
            // This should not happen with the current regex,
            // but as a safeguard, return empty parts if match fails.
            return {
                before: "",
                content: str,
                after: ""
            };
        }
        return {
          before: match[1],
          content: match[2],
          after: match[3]
        };
    }

    function msToString(milliseconds) {
        const totalSeconds = Math.floor(milliseconds / 1000);
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = totalSeconds % 60;
        if (minutes > 0) {
            return `${minutes}m${seconds}s`;
        } else {
            return `${seconds}s`;
        }
    }

    // ---

    /**
     * Animates-in an element.
     * Assumes the existence of a css transition on the element, and a "showing" css class.
     * param isFlex - applies display:flex, else display:block
     */
    function showElement(element, displayValue = "block") {
        element.removeEventListener('transitionend', onHideElementDisplayNone);
        element.style.display = displayValue;
        element.offsetHeight; // Force reflow
        element.classList.add("showing");
    }

    /**
     * Animates-out an element.
     */
    function hideElement(element, shouldDisplayNone=true, now=false) {
        element.removeEventListener('transitionend', onHideElementDisplayNone);
        if (shouldDisplayNone) {
            element.addEventListener('transitionend', onHideElementDisplayNone, { once: true } );
        }
        element.classList.remove("showing");
    }

    function onHideElementDisplayNone(e) {
        e.currentTarget.style.display = "none";
    }

    function isElementShowing(element) {
        return element.classList.contains("showing");
    }

    function isMouseOverElement(element) {
        const els = document.elementsFromPoint(mousePosition.x, mousePosition.y);
        return (els.indexOf(element) > -1);
    }
};

const cl = console.log;

const CORS_GITHUB_URL = "https://zeropointnine.github.io"













/**
 * Generates a synchronous 32-bit hash of an object
 */
function getObjectHashSync(obj) {
 
    // Stable string representation of object
    const str = JSON.stringify(obj, Object.keys(obj).sort());

    // FNV-1a Hashing Algorithm
    let hash = 2166136261; // FNV offset basis
    for (let i = 0; i < str.length; i++) {
        hash ^= str.charCodeAt(i);
        // Perform 32-bit integer multiplication
        hash = Math.imul(hash, 16777619); // FNV prime
    }
    // Convert to unsigned hex string
    return (hash >>> 0).toString(16);
}