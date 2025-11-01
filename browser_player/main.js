"use strict";

window.app = function() {

    const SLEEP_MS = (1000 * 60) * 15

    const root = document.documentElement;
    const loadFileInput = document.getElementById('loadFileInput');
    const loadLastHolder = document.getElementById("loadLast")
    const sleepTimeLeft = document.getElementById('sleepTimeLeft');
    const fileNameDiv = document.getElementById('fileName')
    const playerOverlay = document.getElementById('playerOverlay'); // always visible
    const playerHolder = document.getElementById('playerHolder'); // animates in and out
    const audio = document.getElementById('audio');
    const textHolder = document.getElementById('textHolder');
    const loadLocalButtonLabel = document.getElementById("loadLocalButtonLabel");
    const loadUrlInput = document.getElementById('loadUrlInput');
    const helpHolder = document.getElementById("helpHolder");
    const githubButton = document.getElementById("githubCorner");

    const uiPanelButton = document.getElementById('uiPanelButton');
    const uiOverlay = document.getElementById("uiOverlay");
    const uiPanel = document.getElementById("uiPanel");
    const toast = document.getElementById("toast");
    const loadingOverlay = document.getElementById("loadingOverlay")

    const scrollTopButton = document.getElementById('scrollTopButton')
    const textSizeButton = document.getElementById('textSizeButton')
    const themeButton = document.getElementById('themeButton');
    const segmentColorsButton = document.getElementById('segmentColorsButton')
    const sleepButton = document.getElementById("sleepButton");

    let player = null;
    let file = null;
    let url = null;

    let fileId = null; // id used to track the loaded resource

    let isCheckingZombie = false;

    let timedTextSegments = [];
    let spans = []; // cached text segment span refs

    let isStarted = false; // is audiobook populated and audio loaded
    let hasPlayedOnce = false; // has audio playback happened at least once
    let isPlayerHover = false;
    let isPlayerFocused = false;
    let isToastPinned = false;

    let currentIndex = -1; // segment index whose time range encloses the player's currentTime
    let previousIndex = -1;
    let directSelections = []

    let playerHideDelayId = -1;
    let toastHideDelayId = -1
    let sleepId = -1;
    let sleepEndTime = -1;
    let loopIntervalId = -1;

    // ****
    init();
    // ****

    function init() {

        player = new AudioPlayer({ audioElement: audio, container: playerHolder });

        if (hasPersistentKeyboard()) {
            helpHolder.style.display = "block";
        }

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
            hasPlayedOnce = true;
            root.setAttribute("data-player-status", "play");
            currentIndex = -1; // ensures scroll to current segment
        });

        audio.addEventListener('pause', function() {
            root.setAttribute("data-player-status", "pause");
        });

        audio.addEventListener('error', function(e) {
            // error = e.target.error
            // console.log("Audio error:", error.code, error.message)
        });

        player.playPauseBtn.addEventListener('click', function(e) {
            console.log('hello', this);
            if (isToastPinned) {
                hideToast();
            }
        });

        initPlayerOverlayEvents();

        textHolder.addEventListener('click', onTextClick);

        document.addEventListener("keydown", onKeyDown);

        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') {
                // Browser tab has gone from not-visible to visible
                // If it's a local file, audio is paused, and is past 0s, check for 'zombie state'
                const should = (!isCheckingZombie && !audio.ended && audio.paused && audio.currentTime > 0)
                if (should) {
                    checkZombieState();
                }
            }
        });

        // Load file from queryparam "url"
        const urlParams = new URLSearchParams(window.location.search);
        const url = urlParams.get('url');
        if (url) {
            loadAudioFileOrUrl(null, url);
        }

        uiPanelButton.addEventListener('click', (e) => {
            collapseOptionsButton();
            if (!isElementShowing(uiOverlay)) {
                showUiPanel();
            } else {
                hideUiPanel();
            }
        });

        const lastFileId = localStorage.getItem("last_file_id")
        if (lastFileId) {
            document.getElementById("loadLastId").textContent = lastFileId;
            loadLastHolder.style.display = "block";
        }

        initRootAttributesFromLocalStorage();
        initUiPanelButtons();
    }

    function initRootAttributesFromLocalStorage() {
        const keys = Object.keys(localStorage);
        keys.forEach(key => {
            const value = localStorage[key]
            if (key.startsWith("data-")) {
                root.setAttribute(key, value)
            }
        });
    }

    function initUiPanelButtons() {

        uiOverlay.addEventListener("click", (e) => {
            e.stopPropagation();
            hideUiPanel()
        });

        uiPanel.addEventListener("click", (e) => {
            e.stopPropagation()
        });

        // Scroll to top
        scrollTopButton.addEventListener('click', (e) => {
            e.stopPropagation();
            hideUiPanel();
            window.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
        });

        // Text size
        textSizeButton.addEventListener('click', (e) => {
            e.stopPropagation();
            cycleRootAttribute("data-book-text-size", ["medium", "small"]);
            updateUiPanelButtons()
        });

        // Color theme
        themeButton.addEventListener('click', (e) => {
            e.stopPropagation();
            cycleRootAttribute("data-theme", ["dark"])
            updateUiPanelButtons()
        });

        // Segment colors
        segmentColorsButton.addEventListener('click', (e) => {
            e.stopPropagation();
            cycleRootAttribute("data-segment-colors", ["blue", "red"])
            updateUiPanelButtons()
        });

        // Sleep
        sleepButton.addEventListener("click", (e) => {
            e.stopPropagation();
            const value = cycleRootAttribute("data-sleep", ["on"], false);
            if (value == "on") {
                startSleep();
            } else {
                clearSleep()
            }
            updateUiPanelButtons()
            hideUiPanel()
        });
    }

    /**
     * pFile and pUrl are mutually exclusive
     */
    async function loadAudioFileOrUrl(pFile, pUrl) {

        if (pUrl) {
            showElement(loadingOverlay, true);
        } else {
            // Not rly necessary for local file
        }

        const appMetadata = await loadAppMetadata(pFile, pUrl);

        if (pUrl) {
            hideElement(loadingOverlay);
        }

        if (!appMetadata || typeof appMetadata === 'string') {
            const errorMessage = appMetadata || "No tts-audiobook-tool metadata found";
            alert(errorMessage);
            return;
        }

        start(pFile, pUrl, appMetadata["text_segments"]);
    }

    /**
     * pFile and pUrl are mutually exclusive
     */
    function start(pFile, pUrl, pTimedTextSegments) {

        file = pFile;
        url = pUrl;
        fileId = file ? file.name : url
        timedTextSegments = pTimedTextSegments

        localStorage.setItem("last_file_id", fileId);

        currentIndex = -1;
        previousIndex = -1

        audio.src = "";
        audio.load(); // aborts any pending activity
        hidePlayer();
        fileNameDiv.style.display = "none"
        textHolder.style.display = "none";
        root.setAttribute("data-player-status", "none");
        githubButton.style.display = "block";
        loadingOverlay.style.display = 'none';

        githubButton.style.display = "none";
        loadLastHolder.style.display = "none";
        helpHolder.style.display = "none";
        fileNameDiv.style.display = "block"
        fileNameDiv.textContent = file ? file.name : url

        populateText()


        if (hasPlayedOnce) {
            showAndHidePlayer();
        } else {
            // Don't auto-hide player on first audio load
            showPlayer();
        }

        audio.src = url || URL.createObjectURL(file)
        playerPlay();

        let time = localStorage.getItem("fileId_" + fileId)
        if (time) {
            time = parseFloat(time);
            if (time) {
                audio.currentTime = time;
            }
        }

        if (document.activeElement && document.activeElement.blur) {
            document.activeElement.blur();
        }

        if (isTouchDevice()) {
            // On touch device, player is always showing
            playerHolder.classList.add("showing");
        }

        loopIntervalId = setInterval(loop, 50);
        isStarted = true;
    }

    function populateText() {

        let contentHtml = '';
        timedTextSegments.forEach((segment, i) => {
            const o = splitWhitespace(segment.text)
            if (o["before"]) {
                contentHtml += escapeHtml( o["before"] );
            }
            contentHtml += `<span id="segment-${i}">${ escapeHtml( o["content"] ) }</span>`;
            if (o["after"]) {
                contentHtml += escapeHtml( o["after"] );
            }
        });

        textHolder.innerHTML = contentHtml;
        textHolder.style.display = "block";

        // Cache span references
        spans = [];
        for (let i = 0; i < timedTextSegments.length; i++) {
            spans[i] = document.getElementById("segment-" + i);
        }
    }

    function updateUiPanelButtons() {
        setButtonChildVisible(textSizeButton, root.getAttribute("data-book-text-size"));
        setButtonChildVisible(themeButton, root.getAttribute("data-theme"));
        setButtonChildVisible(sleepButton, root.getAttribute("data-sleep"));
        // rem, segment colors button does not multiple children
    }

    function showUiPanel() {
        updateUiPanelButtons();
        document.body.classList.add('bodyNoScroll'); // Prevent background scrolling
        showElement(uiOverlay)
    }

    function hideUiPanel() {
        document.body.classList.remove('bodyNoScroll');
        hideElement(uiOverlay);
    }

    /**
     * Show toast and hides it after a delay
     */
    function showToast(message, isPinned=false) {
        toast.textContent = message;
        showElement(toast);
        isToastPinned = isPinned;
        clearTimeout(toastHideDelayId);
        if (!isToastPinned) {
            toastHideDelayId = setTimeout(() => { hideToast(); }, 2500);
        }
    }

    function hideToast() {
        isToastPinned = false;
        hideElement(toast);
    }

    // --------------------------------------

    function onKeyDown(event) {

        if (event.target.tagName == "INPUT") {
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
        }
    }

    function onTextClick(event) {

        const clickedSpan = event.target;

        const isSegment = (clickedSpan.tagName === 'SPAN' && clickedSpan.id.startsWith('segment-'));
        if (!isSegment) {
            return;
        }

        const segmentIndex = parseInt(clickedSpan.id.split('-')[1]);
        const segment = timedTextSegments[segmentIndex];

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
            showAndHidePlayer();

            const isSameAsLast = (directSelections.at(-1) !== undefined) && (directSelections.at(-1) == segmentIndex)
            if (!isSameAsLast) {
                directSelections.push(segmentIndex);
            }
        }
    }

    function loop() {

        if (isCheckingZombie) {
            return;
        }
        const i = getSegmentIndexBySeconds(audio.currentTime);
        if (i == currentIndex) {
            return;
        }

        // Index has changed:

        previousIndex = currentIndex;
        const previousSpan = getSpanByIndex(previousIndex);
        currentIndex = i;

        // Unhighlight previous
        previousSpan?.classList.remove('highlight');

        // Highlight active and scroll-to
        if (currentIndex >= 0) {
            getCurrentSpan().classList.add('highlight');
            if (!(document.activeElement instanceof HTMLInputElement)) {
                getCurrentSpan().scrollIntoView({
                    behavior: 'smooth',
                    block: 'center', // 'start', 'center', 'end', or 'nearest'
                    inline: 'nearest' // 'start', 'center', 'end', or 'nearest'
                });
            }
            if (!audio.ended) {
                storePosition();
            }
        }
    }

    // -------------------------------------

    function storePosition(value) {
        value = value || audio.currentTime;
        localStorage.setItem("fileId_" + fileId, value);
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
        // TODO: should have "startFromIndex" and fans out from there
        for (let i = 0; i < timedTextSegments.length; i++) {
            const segment = timedTextSegments[i];
            if (seconds >= segment["time_start"] && seconds < segment["time_end"]) {
                return i
            }
        }
        return -1
    }

    /**
     * Seeks to the next segment that has a starting time
     */
    function seekNextSegment() {
        let index = currentIndex;
        if (index == -1) {
            index = previousIndex;
        }

        for (let i = index + 1; i <= index + 100; i++) {
            if (i >= timedTextSegments.length) {
                return;
            }
            const segment = timedTextSegments[i]
            const has_time = (segment["time_end"] > 0);
            if (has_time) {
                seekBySegmentIndex(i);
                return;
            }
        }
    }

    function seekPreviousSegment() {
        let index = currentIndex;
        if (index == -1) {
            index = previousIndex + 1; // nb +1
        }

        for (let i = index - 1; i >= index - 100; i--) {
            if (i < 0) {
                break;
            }
            const segment = timedTextSegments[i]
            const has_time = (segment["time_end"] > 0);
            if (has_time) {
                seekBySegmentIndex(i);
                break;
            }
        }
    }

    function seekBySegmentIndex(i, andPlay) {
        getCurrentSpan()?.classList.remove('highlight');
        const targetTime = timedTextSegments[i]["time_start"];
        audio.currentTime = targetTime;
        if (audio.paused && andPlay) {
            playerPlay();
        }
    }

    async function playerPlay() {
        try {
            await audio.play();
            if (!getPlayerActive()) {
                showAndHidePlayer();
            }
            if (isToastPinned) {
                hideToast();
            }
        } catch (error) {
            if (error.name == "NotAllowedError") {
                showToast("Click Play to start", true);
            } else {
                console.error("playerPlay error - code:", error.code, "name:", error.name, "message:", error.message)
            }
        }
    }

    function startSleep() {
        showToast("Sleep timer: Will auto-pause in 15 minutes");

        sleepEndTime = new Date().getTime() + SLEEP_MS;

        clearInterval(sleepId);
        onSleepInterval();
        sleepId = setInterval(onSleepInterval, 1000);
    }

    function clearSleep(isFinishedMessage=false) {
        const message = isFinishedMessage ? "Sleep timer finished" : "Sleep timer cancelled";
        showToast(message);

        root.setAttribute("data-sleep", "");
        clearInterval(sleepId);
        updateUiPanelButtons();
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
     * Checks if <audio> is in a 'zombie state'
     * (no longer has a 'handle' on the data but reports no issues)
     */
    function checkZombieState() {

        // Resume playback briefly and see if it jumps to an "ended" state (yes rly)

        isCheckingZombie = true;
        let timeoutId = 0;
        const originalVolume = audio.volume;
        const originalCurrentTime = audio.currentTime;
        audio.volume = 0.0;

        // Minor FYI: Will fail if source is url and has not yet been user-activated
        audio.play();

        const cleanUp = () => {
            isCheckingZombie = false;
            audio.removeEventListener("ended", onEnded);
            clearTimeout(timeoutId);
            audio.pause();
            audio.volume = originalVolume;
        };

        const onEnded = () => {

            if (file) {

                // Unrecoverable. Requires user intervention (ie, click on <input type="file">)
                let s = "The browser has dropped the handle to the local audio file.\n\n";
                s += "This can occur when the mobile browser tab is put into the background while the audio is paused, unfortunately.\n\n";
                s += "Please load the file again to resume.";
                alert(s);
                cleanUp();
                window.location.reload();

            } else {

                // Is url
                // Note: I've not been able to induce zombie state while using a url,
                // but have gotten reports that it does happen on iOS.

                // TODO: untested. verify:
                console.log("restoring connection; unverified");
                cleanUp();
                audio.url = url;
                audio.load(); // for good measure
                audio.currentTime = originalCurrentTime;
            }
        };

        const onTimeout = () => {
            // success
            cleanUp();
        };

        audio.addEventListener("ended", onEnded,{ once: true } );
        timeoutId = setTimeout(onTimeout, 1000);
    }

    // --------------------------------------
    // Player show/hide logic

    function initPlayerOverlayEvents() {
        playerOverlay.addEventListener('mouseenter', () => {
            isPlayerHover = true;
            onPlayerIxChange();
        });

        playerOverlay.addEventListener('mouseleave', () => {
            isPlayerHover = false;
            onPlayerIxChange();
        });

        // Focus in/out detection (bubbles)
        // TODO: new player doesn't support focus atm. revisit.
        playerOverlay.addEventListener('focusin', () => {
            isPlayerFocused = true;
            onPlayerIxChange();
        });
        playerOverlay.addEventListener('focusout', () => {
            setTimeout(() => { // nb, required
            if (!playerOverlay.contains(document.activeElement)) {
                isPlayerFocused = false;
                onPlayerIxChange();
            }
            }, 0);
        });
    }

    function onPlayerIxChange() {
        // Should be called when:
        // Mouse has either entered or left player holder area
        // Focus has entered or left thhe player holder and children
        if (getPlayerActive()) {
            clearTimeout(playerHideDelayId);
            showPlayer();
        } else {
            showAndHidePlayer(0);
        }
    }

    function showAndHidePlayer(hideDelay=1000) {
        if (isTouchDevice()) {
            return;
        }
        showPlayer();
        playerHideDelayId = setTimeout(() => { hidePlayer(); }, hideDelay);
    }

    function showPlayer() {
        if (isTouchDevice()) {
            return;
        }
        clearTimeout(playerHideDelayId);
        showElement(playerHolder);
    }

    function hidePlayer() {
        if (isTouchDevice()) {
            return;
        }
        clearTimeout(playerHideDelayId);
        hideElement(playerHolder, false);
    }

    function getPlayerActive() {
        return isPlayerFocused || isPlayerHover;
    }

    // --------------------------------------
    // 'Utility' functions
    // TODO: move to dedicated script after implementing script concat

    function isTouchDevice() {
        // hand-wavey
        return !matchMedia('(pointer:fine)').matches
    }

    function hasPersistentKeyboard() {
        // hand-wavey test
        const hasFinePointer = window.matchMedia('(pointer: fine)').matches;
        const hasHoverSupport = window.matchMedia('(hover: hover)').matches;
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
    function showElement(element, isFlex) {
        element.removeEventListener('transitionend', onHideElementDisplayNone);
        element.style.display = isFlex ? "flex" : "block";
        element.offsetHeight; // Force reflow
        element.classList.add("showing");
    }

    /**
     * Animates-out an element.
     * param now - skips transition
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
};
