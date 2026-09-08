import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import test from "node:test";

const repoRoot = fileURLToPath(new URL("../", import.meta.url));
const context = vm.createContext({ console });
const utilSource = readFileSync(`${repoRoot}/browser_player/util.js`, "utf8");
vm.runInContext(`${utilSource}\nglobalThis.Util = Util;`, context);
const bookTextSource = readFileSync(`${repoRoot}/browser_player/book-text.js`, "utf8");
vm.runInContext(`${bookTextSource}\nglobalThis.BookText = BookText;`, context);

const BookText = context.BookText;
const segment = (text, timeStart, timeEnd) => ({
    text,
    time_start: timeStart,
    time_end: timeEnd,
    playable: timeEnd > timeStart,
});
const plain = (value) => JSON.parse(JSON.stringify(value));

function makeLookupBook(textSegments, textSegmentGroups = []) {
    const bookText = Object.create(BookText.prototype);
    bookText._populateText = () => {};
    bookText.init(textSegments, false, [], textSegmentGroups);
    return bookText;
}

test("time lookup handles exact boundaries, gaps, and zero-timed leaves", () => {
    const bookText = makeLookupBook([
        segment("One.", 0, 1),
        segment("Missing.", 0, 0),
        segment("Two.", 2, 3),
    ]);

    assert.equal(bookText.isAudioTimelineOrdered, true);
    assert.equal(bookText.getSegmentIndexBySeconds(0), 0);
    assert.equal(bookText.getSegmentIndexBySeconds(0.999), 0);
    assert.equal(bookText.getSegmentIndexBySeconds(1), -1);
    assert.equal(bookText.getSegmentIndexBySeconds(1.5), -1);
    assert.equal(bookText.getSegmentIndexBySeconds(2), 2);
    assert.equal(bookText.getSegmentIndexBySeconds(3), -1);
});

test("time lookup falls back safely for non-monotonic and overlapping leaves", () => {
    const nonMonotonic = makeLookupBook([
        segment("Later.", 2, 3),
        segment("Earlier.", 0, 1),
    ]);
    assert.equal(nonMonotonic.isAudioTimelineOrdered, false);
    assert.equal(nonMonotonic.getSegmentIndexBySeconds(0.5), 1);
    assert.equal(nonMonotonic.getSegmentIndexBySeconds(2.5), 0);

    const overlapping = makeLookupBook([
        segment("First.", 0, 2),
        segment("Second.", 1, 3),
    ]);
    assert.equal(overlapping.isAudioTimelineOrdered, false);
    assert.equal(overlapping.getSegmentIndexBySeconds(1.5), 0);
    overlapping.currentIndex = 1;
    assert.equal(overlapping.getSegmentIndexBySeconds(1.5), 1);
});

test("section rendering falls back when a boundary bisects a group", () => {
    const bookText = makeLookupBook(
        [
            segment("Flat.", 0, 1),
            segment("Nested one.", 1, 2),
            segment("Nested two.", 2, 3),
            segment("Tail.", 3, 4),
        ],
        [{ groupIndex: 0, startIndex: 1, endIndex: 3, leafCount: 2 }],
    );
    const ranges = bookText._getDisplaySectionRanges([
        { start_index: 0, end_index: 2 },
        { start_index: 2, end_index: 4 },
    ]);

    assert.deepEqual(plain(ranges), [{ startIndex: 0, endIndex: 4 }]);
});

test("ordered lookup remains fast and correct for a large leaf sequence", () => {
    const leafCount = 50000;
    const leaves = Array.from(
        { length: leafCount },
        (_, index) => segment(`Leaf ${index}`, index, index + 1),
    );
    const bookText = makeLookupBook(leaves);

    assert.equal(bookText.isAudioTimelineOrdered, true);
    assert.equal(bookText.getSegmentIndexBySeconds(leafCount - 0.5), leafCount - 1);
});
