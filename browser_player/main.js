window.app = function() {

    const DEFAULT_FADE_DELAY = 1000;
    const SLEEP_DURATION = 1000 * 60 * 15;
    const DEMO_URL_A = "waves-oute"
    const DEMO_URL_B = "waves-chatterbox"

    const root = document.documentElement;
    const helpHolder = document.getElementById("helpHolder");
    const loadFileInput = document.getElementById('loadFileInput');
    const sleepTimeLeft = document.getElementById('sleepTimeLeft');
    const fileNameDiv = document.getElementById('fileName')
    const playerHolder = document.getElementById('playerHolder');
    const player = document.getElementById('player');
    const textHolder = document.getElementById('textHolder');
    const loadLocalButtonLabel = document.getElementById("loadLocalButtonLabel");
    const loadUrlInput = document.getElementById('loadUrlInput');

    const uiPanelButton = document.getElementById('uiPanelButton');
    const uiOverlay = document.getElementById("uiOverlay");
    const uiPanel = document.getElementById("uiPanel")

    const scrollTopButton = document.getElementById('scrollTopButton')
    const textSizeButton = document.getElementById('textSizeButton')
    const themeButton = document.getElementById('themeButton');
    const segmentColorsButton = document.getElementById('segmentColorsButton')
    const sleepButton = document.getElementById("sleepButton");

    let fileId = null;
    let timedTextSegments = [];
    let spans = []; // cached text segment spans array

    let isStarted = false;
    let isPlayerHover = false;
    let isPlayerFocused = false;

    let currentIndex = -1; // segment index whose time range encloses the player's currentTime
    let previousIndex = -1;

    let intervalId = -1;
    let fadeOutId = -1;
    let sleepId = -1;
    let sleepEndTime = -1;
    let fadeOutValue = 0.0;
    let lastSavePositionTime = 0

    // ****
    init();
    // ****

    function init() {

        if (hasPersistentKeyboard()) {
            helpHolder.style.display = "block";
        }

        if (isTouchDevice()) {
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
            currentIndex = -1;
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

        uiPanelButton.addEventListener('click', (e) => {
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
        start(fileOrUrl, result["text_segments"]);
    }

    function clear() {
        isStarted = false;
        playerHolder.style.display = "none";
        fileNameDiv.style.display = "none"
        textHolder.style.display = "none";
        clearInterval(intervalId)
        player.src = null;
        currentIndex = -1;
        previousIndex = -1
        spans = [];
        root.setAttribute("data-player-status", "none");
    }

    function start(fileOrUrl, pTimedTextSegments) {

        let file = null;
        let url = null;
        if (typeof fileOrUrl === "string") {
            url = fileOrUrl;
            fileId = url;
        } else {
            file = fileOrUrl;
            fileId = file.name;
        }

        isDemoUrl = url && (url.includes(DEMO_URL_A) || url.includes(DEMO_URL_B));
        if (!isDemoUrl) {
            document.getElementById("githubCorner").style.display = "none";
        }

        timedTextSegments = pTimedTextSegments

        helpHolder.style.display = "none";

        fileNameDiv.style.display = "block"
        fileNameDiv.textContent = file ? file.name : url

        populateText()

        playerHolder.style.display = "block";
        showPlayerAndFade();

        player.src = file ? URL.createObjectURL(file) : url
        playerPlay();

        time = localStorage.getItem("fileId_" + fileId)
        if (time) {
            time = parseFloat(time);
            if (time) {
                player.currentTime = time;
            }
        }

        if (document.activeElement && document.activeElement.blur) {
            document.activeElement.blur();
        }

        intervalId = setInterval(loop, 50);
        isStarted = true;
    }

    function populateText() {

        let contentHtml = '';
        timedTextSegments.forEach((segment, i) => {

            // contentHtml += `<span id="segment-${i}">${segment.text}</span>`;

            o = splitWhitespace(segment.text)
            if (o["before"]) {
                contentHtml += o["before"];
            }
            contentHtml += `<span id="segment-${i}">${o["content"]}</span>`;
            if (o["after"]) {
                contentHtml += o["after"];
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

        const doLoadLocal = function() {
            loadFileInput.click();
            event.preventDefault();
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

        // Hotkeys that are active while audio is loaded
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
            case ",":
                player.currentTime -= 60;
                break;
            case ".":
                player.currentTime += 60;
                break;
        }
    }

    function onTextClick(event) {

        const clickedSpan = event.target;

        isSegment = (clickedSpan.tagName === 'SPAN' && clickedSpan.id.startsWith('segment-'));
        if (!isSegment) {
            return;
        }

        const segmentIndex = parseInt(clickedSpan.id.split('-')[1]);
        const segment = timedTextSegments[segmentIndex]

        if (segment["time_start"] == 0 && segment["time_end"] == 0) {
            return
        }

        if (clickedSpan == getCurrentSpan()) {
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

    function loop() {

        // Save currentTime every 10 seconds
        if (Date.now() - lastSavePositionTime > 10000) {
            localStorage.setItem("fileId_" + fileId, player.currentTime);
            lastSavePositionTime = Date.now()
        }

        const i = getSegmentIndexBySeconds(player.currentTime);
        if (i == currentIndex) {
            // Update only when index has changed
            return;
        }
        previousIndex = currentIndex;
        const previousSpan = getSpanByIndex(currentIndex);
        currentIndex = i;

        // Unhighlight previous
        previousSpan?.classList.remove('highlight');

        // Highlight active and scroll-to
        if (currentIndex >= 0) {
            getCurrentSpan().classList.add('highlight');
            getCurrentSpan().scrollIntoView({
                behavior: 'smooth',
                block: 'center', // 'start', 'center', 'end', or 'nearest'
                inline: 'nearest' // 'start', 'center', 'end', or 'nearest'
            });
            localStorage.setItem("fileId_" + fileId, player.currentTime);
            lastSavePositionTime = Date.now()
        }

        // Save currentTime
        localStorage.setItem("fileId_" + fileId, player.currentTime);
    }

    // -------------------------------------

    function getSpanByIndex(i) {
        return i >= 0 ? spans[i] : null;
    }

    function getCurrentSpan() {
        return currentIndex >= 0 ? spans[currentIndex] : null;
    }

    function getSegmentIndexBySeconds(seconds) {
        // TODO: should have "startFromIndex" and fans out from there
        for (let i = 0; i < timedTextSegments.length; i++) {
            segment = timedTextSegments[i];
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
            segment = timedTextSegments[i]
            has_time = (segment["time_end"] > 0);
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
            segment = timedTextSegments[i]
            has_time = (segment["time_end"] > 0);
            if (has_time) {
                seekBySegmentIndex(i);
                break;
            }
        }
    }

    function seekBySegmentIndex(i) {
        getCurrentSpan()?.classList.remove('highlight');

        targetTime = timedTextSegments[i]["time_start"];
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

};
