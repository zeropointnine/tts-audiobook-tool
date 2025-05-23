(function() {

    let rawText = "";
    let timedTextSegments = [];

    let flacFileInput = null;
    let fileNameDiv = null;
    let audioPlayer = null;
    let textHolder = null;
    let themeButton = null;
    let loadUrlInput = null;

    let selectedSpan = null;
    let intervalId = -1;

    function init() {

        flacFileInput = document.getElementById('flacFileInput');
        fileNameDiv = document.getElementById('fileName')
        audioPlayer = document.getElementById('audioPlayer');
        textHolder = document.getElementById('textHolder');
        themeButton = document.getElementById('themeButton');
        loadUrlInput = document.getElementById('loadUrlInput');

        loadUrlInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                const url = loadUrlInput.value.trim()
                if (url) {
                    loadUrlInput.blur();
                    loadFlac(url);
                }
            }
        });

        loadUrlInput.addEventListener('blur', () => {
            loadUrlInput.value = '';
        });

        flacFileInput.addEventListener('change', async () => {
            const file = flacFileInput.files[0];
            if (file) {
                loadFlac(file);
            }
        });

        audioPlayer.addEventListener('play', function() {
            selectedSpan = null; // ensures scroll to current audio segment
        });

        textHolder.addEventListener('click', (event) => {
            const target = event.target;
            if (target.tagName === 'SPAN' && target.id.startsWith('segment-')) {
                const segmentIndex = parseInt(target.id.split('-')[1]);
                if (!isNaN(segmentIndex) && timedTextSegments[segmentIndex]) {
                    if (selectedSpan && selectedSpan != target) {
                        selectedSpan.classList.remove('highlight');
                    }
                    audioPlayer.currentTime = timedTextSegments[segmentIndex].time_start;
                    audioPlayer.play();
                }
            }
        });

        // Color theme toggle
        const html = document.documentElement;
        themeButton.addEventListener('click', () => {
            if (html.getAttribute('data-theme') === 'dark') {
                html.removeAttribute('data-theme');
                localStorage.setItem('darkMode', 'false');
            } else {
                html.setAttribute('data-theme', 'dark');
                localStorage.setItem('darkMode', 'true');
            }
        });
        if (localStorage.getItem('darkMode') === 'true') {
            html.setAttribute('data-theme', 'dark');
        }

        // When the queryparam is "url", run the function "loadFlac" using the value.
        const urlParams = new URLSearchParams(window.location.search);
        const url = urlParams.get('url');
        if (url) {
            loadFlac(url);
        }
    }

    async function loadFlac(fileOrUrl) {
        clear();

        result = await loadMetadataFromAppFlac(fileOrUrl);
        if (!result) {
            alert("No tts-audiobook-tool metadata found");
            return;
        }
        initPage(fileOrUrl, result["raw_text"], result["text_segments"]);
    }

    function clear() {
        clearInterval(intervalId)
        audioPlayer.src = null;
        audioPlayer.style.display = "none";
        fileNameDiv.style.display = "none"
        textHolder.style.display = "none";
        selectedSpan = null
    }

    function initPage(fileOrUrl, pRawText, pTimedTextSegments) {

        let file = null;
        let url = null;
        if (typeof fileOrUrl === "string") {
            url = fileOrUrl;
        } else {
            file = fileOrUrl;
        }

        // if (url) {
        //     const newUrl = new URL(window.location.href);
        //     newUrl.searchParams.set('url', url);
        //     window.history.pushState({ url: url }, document.title, newUrl.toString());
        // }

        rawText = pRawText
        timedTextSegments = pTimedTextSegments

        fileNameDiv.style.display = "block"
        fileNameDiv.textContent = file ? file.name : url

        populateText()

        audioPlayer.src = file ? URL.createObjectURL(file) : url
        audioPlayer.play();
        audioPlayer.style.display = "block";

        intervalId = setInterval(loop, 50)
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

        const seconds = audioPlayer.currentTime

        let span = null;
        for (let i = 0; i < timedTextSegments.length; i++) {
            const segment = timedTextSegments[i];
            if (seconds >= segment.time_start && seconds < segment.time_end) {
                span = document.getElementById(`segment-${i}`);
                break; // Found the active segment
            }
        }

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

    window.app = {
        init: init
    };

})();
