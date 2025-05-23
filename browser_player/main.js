(function() {

    let rawText = "";
    let timedTextSegments = [];

    let flacFileInput = null;
    let fileNameDiv = null;
    let audioPlayer = null;
    let textHolder = null;
    let themeButton = null;

    let selectedSpan = null;
    let intervalId = -1;

    function init() {
        flacFileInput = document.getElementById('flacFileInput');
        fileNameDiv = document.getElementById('fileName')
        audioPlayer = document.getElementById('audioPlayer');
        textHolder = document.getElementById('textHolder');
        themeButton = document.getElementById('themeButton');

        flacFileInput.addEventListener('change', async () => {
            clear();
            const file = flacFileInput.files[0];
            if (!file) {
                return;
            }
            result = await loadMetadataFromAppFlac(file);
            if (!result) {
                alert("No tts-audiobook-tool metadata found");
                return;
            }
            initPage(file, result["raw_text"], result["text_segments"]);
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
    }

    function clear() {
        clearInterval(intervalId)
        audioPlayer.src = null;
        audioPlayer.style.display = "none";
        fileNameDiv.style.display = "none"
        textHolder.style.display = "none";
        selectedSpan = null
    }

    function initPage(file, pRawText, pTimedTextSegments) {

        rawText = pRawText
        timedTextSegments = pTimedTextSegments

        fileNameDiv.style.display = "block"
        fileNameDiv.textContent = file.name

        populateText()

        audioPlayer.src = URL.createObjectURL(file);
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
