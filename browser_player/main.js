(function() {

    const DEFAULT_FADE_DELAY = 1000;
    const SLEEP_DURATION = 1000 * 60 * 15;

    const root = document.documentElement;

    let loadFileInput = null;
    let sleepTimeLeft = null;
    let fileNameDiv = null;
    let playerHolder = null;
    let player = null;
    let textHolder = null;
    let loadLocalButtonLabel = null;
    let loadUrlInput = null;

    let uiToggleButton = null;
    let uiOverlay = null;
    let uiPanel = null;

    let scrollTopButton = null;
    let textSizeButton = null;
    let themeButton = null;
    let segmentColorsButton = null;
    let sleepButton = null;

    let rawText = "";
    let timedTextSegments = [];
    let selectedSpan = null;
    let intervalId = -1;
    let fadeOutId = -1;
    let sleepId = -1;
    let sleepEndTime = -1;
    let isStarted = false;
    let isPlayerHover = false;
    let isPlayerFocused = false;
    let fadeOutValue = 0.0;

    function init() {

        loadFileInput = document.getElementById('loadFileInput');
        sleepTimeLeft = document.getElementById('sleepTimeLeft');
        fileNameDiv = document.getElementById('fileName')
        playerHolder = document.getElementById('playerHolder');
        player = document.getElementById('player');
        textHolder = document.getElementById('textHolder');
        loadLocalButtonLabel = document.getElementById("loadLocalButtonLabel");
        loadUrlInput = document.getElementById('loadUrlInput');

        uiToggleButton = document.getElementById('uiToggleButton');
        uiOverlay = document.getElementById("uiOverlay");
        uiPanel = document.getElementById("uiPanel")

        scrollTopButton = document.getElementById('scrollTopButton')
        textSizeButton = document.getElementById('textSizeButton')
        themeButton = document.getElementById('themeButton');
        segmentColorsButton = document.getElementById('segmentColorsButton')
        sleepButton = document.getElementById("sleepButton");

        if (!matchMedia('(pointer:fine)').matches) {
            // Treat as touch device
            // Disable player fadeout
            fadeOutValue = 1.0;
        }

        loadUrlInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                const url = loadUrlInput.value.trim()
                if (url) {
                    loadUrlInput.blur();
                    loadFlacOrMp4(url);
                }
            }
        });

        loadUrlInput.addEventListener('blur', () => {
            loadUrlInput.value = '';
        });

        loadFileInput.addEventListener('change', async () => {
            const file = loadFileInput.files[0];
            if (file) {
                loadFlacOrMp4(file);
            }
        });

        player.addEventListener('play', function() {
            root.setAttribute("data-player-status", "play");
            // ensures scroll to current segment
            selectedSpan = null;
        });

        player.addEventListener('pause', function() {
            root.setAttribute("data-player-status", "pause");
        });

        playerHolder.addEventListener('mouseenter', () => {
            isPlayerHover = true;
            onPlayerIxChange();
        });

        playerHolder.addEventListener('mouseleave', () => {
            isPlayerHover = false;
            onPlayerIxChange();
        });

        // Focus and blur detection (bubbles)
        playerHolder.addEventListener('focusin', () => {
            isPlayerFocused = true;
            onPlayerIxChange();
        });
        playerHolder.addEventListener('focusout', () => {
            setTimeout(() => { // nb, required
            if (!playerHolder.contains(document.activeElement)) {
                isPlayerFocused = false;
                onPlayerIxChange();
            }
            }, 0);
        });

        textHolder.addEventListener('click', onTextClick);

        document.addEventListener("keydown", onKeyDown);

        // When the queryparam is "url", run the function "loadFlacOrMp4" using the value.
        const urlParams = new URLSearchParams(window.location.search);
        const url = urlParams.get('url');
        if (url) {
            loadFlacOrMp4(url);
        }

        uiToggleButton.addEventListener('click', (e) => {
            if (uiOverlay.style.display !== 'block') {
                updateUiPanelButtons();
                uiOverlay.style.display = 'block';
            } else {
                uiOverlay.style.display = 'none';
            }
        });

        initRootAttributesFromLocalStorage();
        initUiPanelButtons();
    }

    function initRootAttributesFromLocalStorage() {
        const keys = Object.keys(localStorage);
        keys.forEach(key => {
            value = localStorage[key]
            if (key.startsWith("data-")) {
                root.setAttribute(key, value)
            }
        });
    }

    function initUiPanelButtons() {

        uiOverlay.addEventListener("click", (e) => {
            e.stopPropagation();
            uiOverlay.style.display = "none";
        });

        uiPanel.addEventListener("click", (e) => {
            e.stopPropagation()
        });

        // Scroll to top
        scrollTopButton.addEventListener('click', (e) => {
            e.stopPropagation();
            window.scrollTo(0, 0);
        });

        // Text size
        textSizeButton.addEventListener('click', (e) => {
            e.stopPropagation();
            cycleRootAttribute("data-text-size", ["medium", "small"]);
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
            cycleRootAttribute("data-segment-colors", ["blue"])
            updateUiPanelButtons()
        });

        // Sleep
        sleepButton.addEventListener("click", (e) => {
            e.stopPropagation();
            value = cycleRootAttribute("data-sleep", ["on"], false);
            if (value == "on") {
                startSleep();
            } else {
                clearSleep()
            }
            updateUiPanelButtons()
        });
    }

    async function loadFlacOrMp4(fileOrUrl) {
        clear();

        result = await loadAppMetadata(fileOrUrl);
        if (!result) {
            alert("No tts-audiobook-tool metadata found");
            return;
        }

        // TEMP
        // result = {
        //     raw_text: "hello",
        //     text_segments: [
        //         {
        //             index_start: 0,
        //             index_end: 4,
        //             time_start: 2.0,
        //             time_end: 3.0
        //         }
        //     ]
        //  }

        start(fileOrUrl, result["raw_text"], result["text_segments"]);
    }

    function clear() {
        isStarted = false;
        playerHolder.style.display = "none";
        fileNameDiv.style.display = "none"
        textHolder.style.display = "none";
        clearInterval(intervalId)
        player.src = null;
        selectedSpan = null;
        root.setAttribute("data-player-status", "none");
    }

    function start(fileOrUrl, pRawText, pTimedTextSegments) {

        let file = null;
        let url = null;
        if (typeof fileOrUrl === "string") {
            url = fileOrUrl;
        } else {
            file = fileOrUrl;
        }

        rawText = pRawText
        timedTextSegments = pTimedTextSegments

        fileNameDiv.style.display = "block"
        fileNameDiv.textContent = file ? file.name : url

        populateText()

        playerHolder.style.display = "block";
        showPlayerAndFade();

        player.src = file ? URL.createObjectURL(file) : url
        playerPlay();

        if (document.activeElement && document.activeElement.blur) {
            document.activeElement.blur();
        }

        intervalId = setInterval(loop, 50);
        isStarted = true;
    }

    function updateUiPanelButtons() {
        setButtonChildVisible(textSizeButton, root.getAttribute("data-text-size"));
        setButtonChildVisible(themeButton, root.getAttribute("data-theme"));
        setButtonChildVisible(sleepButton, root.getAttribute("data-sleep"));
        // rem, segment colors button does not multiple children
    }

    // --------------------------------------

    function onKeyDown(event) {

        if (event.target.tagName == "INPUT") {
            return;
        }
        // console.log(event.key);

        if (event.key === "Enter" || event.key === " ") {
            if (document.activeElement == loadLocalButtonLabel) {
                loadFileInput.click();
                event.preventDefault();
                loadLocalButtonLabel.blur();
            }
            return;
        }

        if (!isStarted) {
            return;
        }
        switch (event.key) {
            case "Escape":
                if (player.paused) {
                    playerPlay();
                } else {
                    player.pause();
                }
                event.preventDefault();
                break;
            case "[":
                seekPreviousSegment();
                break;
            case "]":
                seekNextSegment();
                break;
        }
    }

    function onTextClick(event) {

        isSegment = (event.target.tagName === 'SPAN' && event.target.id.startsWith('segment-'));
        if (!isSegment) {
            return;
        }

        const clickedSpan = event.target;
        const segmentIndex = parseInt(clickedSpan.id.split('-')[1]);

        if (clickedSpan == selectedSpan) {
            // Toggle play
            if (player.paused) {
                playerPlay();
            } else {
                player.pause()
            }
        } else {
            seekBySegmentIndex(segmentIndex);
            showPlayerAndFade();
        }
    }

    function populateText() {

        let contentHtml = '';
        let lastIndex = 0;

        timedTextSegments.forEach((segment, i) => {
            // Add text before the current segment (if any)
            if (segment.index_start > lastIndex) {
                contentHtml += escapeHtml(rawText.substring(lastIndex, segment.index_start));
            }
            // Add the current segment wrapped in a span
            const segmentText = rawText.substring(segment.index_start, segment.index_end);
            contentHtml += `<span id="segment-${i}">${escapeHtml(segmentText)}</span>`;
            lastIndex = segment.index_end;
        });

        // Add any remaining text after the last segment
        if (lastIndex < rawText.length) {
            contentHtml += escapeHtml(rawText.substring(lastIndex));
        }

        textHolder.innerHTML = contentHtml;
        textHolder.style.display = "block";
    }

    function loop() {

        let span = getCurrentSpan()

        // Update highlighting only if the active segment has changed
        if (selectedSpan !== span) {
            if (selectedSpan) {
                selectedSpan.classList.remove('highlight');
            }
            if (span) {
                span.classList.add('highlight');
                // Scroll the new highlighted span into view, centered
                span.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center', // 'start', 'center', 'end', or 'nearest'
                    inline: 'nearest' // 'start', 'center', 'end', or 'nearest'
                });
            }
            selectedSpan = span;
        }
    }

    function seekPreviousSegment() {
        i = getCurrentSegmentIndex()
        if (i == -1) {
            return;
        }
        i = i - 1;
        if (i < 0) {
            return;
        }
        seekBySegmentIndex(i)
    }

    function seekNextSegment() {
        i = getCurrentSegmentIndex()
        if (i == -1) {
            return;
        }
        i = i + 1;
        if (i >= timedTextSegments.length) {
            return;
        }
        seekBySegmentIndex(i)
    }

    function seekBySegmentIndex(i) {
        if (selectedSpan) {
            selectedSpan.classList.remove('highlight');
            selectedSpan = null;
        }

        targetTime = timedTextSegments[i].time_start;
        player.currentTime = targetTime;
        if (player.paused) {
            playerPlay();
        }
    }

    // --------------------------------------
    // Player show/hide logic etc

    async function playerPlay() {
        try {
            await player.play();
            if (!getPlayerActive()) {
                showPlayerAndFade();
            }
        } catch (error) {
            console.error("Playback failed:", error);
        }
    }

    function onPlayerIxChange() {
        // Should be called when:
        // Mouse has either entered or left player holder area
        // Focus has entered or left thhe player holder and children
        if (getPlayerActive()) {
            clearTimeout(fadeOutId);
            playerHolder.style.opacity = '1.0';
        } else {
            showPlayerAndFade(0);
        }
    }

    function showPlayerAndFade(duration) {
        if (isNaN(duration)) {
            duration = DEFAULT_FADE_DELAY;
        }

        playerHolder.classList.add('no-transition');
        playerHolder.style.opacity = '1.0';
        void playerHolder.offsetHeight; // force reflow
        playerHolder.classList.remove('no-transition');

        clearTimeout(fadeOutId);
        fadeOutId = setTimeout(() => {
            playerHolder.style.opacity = fadeOutValue;
        }, duration);
    }

    function getPlayerActive() {
        return isPlayerFocused || isPlayerHover;
    }

    function startSleep() {
        sleepEndTime = new Date().getTime() + SLEEP_DURATION;

        clearInterval(sleepId);
        onSleepInterval();
        sleepId = setInterval(onSleepInterval, 1000);
    }

    function clearSleep() {
        root.setAttribute("data-sleep", "");
        clearInterval(sleepId);
        updateUiPanelButtons();
    }

    function onSleepInterval() {
        const ms = sleepEndTime - new Date().getTime();
        if (ms <= 0) {
            player.pause();
            clearSleep();
            return;
        }
        sleepTimeLeft.textContent = msToString(ms);
    }

    // ----------------------------------------

    function getCurrentSpan() {
        i = getCurrentSegmentIndex()
        if (i == -1) {
            return null
        }
        id = `segment-${i}`
        span = document.getElementById(id);
        return span;
    }

    /**
     * Returns the index of the segment that spans the current play time, or -1.
     */
    function getCurrentSegmentIndex() {
        const seconds = player.currentTime
        for (let i = 0; i < timedTextSegments.length; i++) {
            segment = timedTextSegments[i];
            if (seconds >= segment.time_start && seconds < segment.time_end) {
                return i
            }
        }
        return -1;
    }

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
        value = localStorage.getItem(dataAttribute)
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

            value = child.dataset["value"]

            let isMatch = (value === targetValue);
            isMatch |= (!value || value == "default") && (!targetValue || targetValue == "default");
            if (isMatch) {
                child.style.display = 'revert';
                break;
            }
        }
    }

    window.app = {
        init: init
    };

})();
