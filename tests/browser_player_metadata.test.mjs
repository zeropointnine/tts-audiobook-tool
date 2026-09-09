import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import test from "node:test";

const repoRoot = fileURLToPath(new URL("../", import.meta.url));
const context = vm.createContext({
    console,
    TextDecoder,
    Uint8Array,
    atob: globalThis.atob,
});

const utilSource = readFileSync(`${repoRoot}/browser_player/util.js`, "utf8");
vm.runInContext(`${utilSource}\nglobalThis.Util = Util;`, context);
const metadataSource = readFileSync(`${repoRoot}/browser_player/metadata-util.js`, "utf8");
vm.runInContext(`${metadataSource}\nglobalThis.MetadataUtil = MetadataUtil;`, context);

const MetadataUtil = context.MetadataUtil;
const segment = (text, timeStart, timeEnd) => ({
    text,
    time_start: timeStart,
    time_end: timeEnd,
});
const plain = (value) => JSON.parse(JSON.stringify(value));

test("normalizes legacy flat text segments without groups", () => {
    const result = MetadataUtil.normalizeAppMetadata({
        version: 3,
        text_segments: [segment("One. ", 0, 1), segment("Two.", 1, 2)],
        bookmarks: [1],
        sections: [{ title: "All", start_index: 0, end_index: 2 }],
    });

    assert.notEqual(typeof result, "string");
    assert.deepEqual(plain(result.textSegments.map((item) => item.text)), ["One. ", "Two."]);
    assert.deepEqual(plain(result.textSegmentGroups), []);
    assert.deepEqual(plain(result.sections), [{ title: "All", start_index: 0, end_index: 2 }]);
    assert.deepEqual(plain(result.textSegments.map((item) => item.playable)), [true, true]);
});

test("flattens mixed entries and preserves explicit singleton groups", () => {
    const raw = {
        version: 4,
        text_segments: [
            segment("Flat. ", 0, 1),
            [segment("Nested one, ", 1, 2), segment("nested two. ", 2, 3)],
            [segment("Singleton.", 3, 4)],
        ],
        bookmarks: [0, 2, 3],
        sections: [{ title: "All", start_index: 0, end_index: 4 }],
    };
    const result = MetadataUtil.normalizeAppMetadata(raw);

    assert.notEqual(typeof result, "string");
    assert.deepEqual(
        plain(result.textSegments.map((item) => item.text)),
        ["Flat. ", "Nested one, ", "nested two. ", "Singleton."],
    );
    assert.deepEqual(plain(result.textSegmentGroups), [
        { groupIndex: 0, outerIndex: 1, startIndex: 1, endIndex: 3, leafCount: 2 },
        { groupIndex: 1, outerIndex: 2, startIndex: 3, endIndex: 4, leafCount: 1 },
    ]);
    assert.deepEqual(
        plain(result.textSegments.map(({ flatIndex, outerIndex, childIndex, groupIndex }) => (
            { flatIndex, outerIndex, childIndex, groupIndex }
        ))),
        [
            { flatIndex: 0, outerIndex: 0, childIndex: null, groupIndex: null },
            { flatIndex: 1, outerIndex: 1, childIndex: 0, groupIndex: 0 },
            { flatIndex: 2, outerIndex: 1, childIndex: 1, groupIndex: 0 },
            { flatIndex: 3, outerIndex: 2, childIndex: 0, groupIndex: 1 },
        ],
    );
    assert.deepEqual(plain(result.sections), [{ title: "All", start_index: 0, end_index: 4 }]);
});

test("flat and nested topology have the same persisted identity", () => {
    const leaves = [segment("One. ", 0, 1), segment("Two.", 1, 2)];
    const flat = MetadataUtil.normalizeAppMetadata({ text_segments: leaves });
    const nested = MetadataUtil.normalizeAppMetadata({ text_segments: [[...leaves]] });

    assert.notEqual(typeof flat, "string");
    assert.notEqual(typeof nested, "string");
    assert.equal(flat.identity.textId, nested.identity.textId);
    assert.equal(flat.identity.positionId, nested.identity.positionId);
});

test("validates sections against flattened leaf count", () => {
    const result = MetadataUtil.normalizeAppMetadata({
        text_segments: [[segment("One. ", 0, 1), segment("Two.", 1, 2)]],
        sections: [
            { title: "Valid", start_index: 0, end_index: 2 },
            { title: "Too far", start_index: 0, end_index: 3 },
        ],
    });

    assert.notEqual(typeof result, "string");
    assert.deepEqual(plain(result.sections), [{ title: "Valid", start_index: 0, end_index: 2 }]);
});

test("rejects metadata versions newer than the player supports", () => {
    const tooNew = {
        version: MetadataUtil.MAX_SUPPORTED_ABR_VERSION + 1,
        text_segments: "some future shape that would fail validation",
    };
    const result = MetadataUtil.normalizeAppMetadata(tooNew);

    assert.equal(typeof result, "string");
    assert.match(result, /newer than this player supports/);
    assert.match(result, new RegExp(`version ${MetadataUtil.MAX_SUPPORTED_ABR_VERSION}`));
});

test("accepts metadata at the maximum supported version", () => {
    const result = MetadataUtil.normalizeAppMetadata({
        version: MetadataUtil.MAX_SUPPORTED_ABR_VERSION,
        text_segments: [segment("One. ", 0, 1)],
    });

    assert.notEqual(typeof result, "string");
    assert.equal(result.version, MetadataUtil.MAX_SUPPORTED_ABR_VERSION);
});

test("rejects malformed nested and timed-segment entries", () => {
    const invalidCases = [
        { raw: [], expected: "missing required field" },
        { raw: [[]], expected: "must not be an empty list" },
        { raw: [[[segment("Deep.", 0, 1)]]], expected: "must be a timed segment object" },
        { raw: [[null]], expected: "must be a timed segment object" },
        { raw: [segment(123, 0, 1)], expected: ".text must be a string" },
        { raw: [segment("Bad start.", "0", 1)], expected: ".time_start must be a finite number" },
        { raw: [segment("Bad end.", 0, Number.NaN)], expected: ".time_end must be a finite number" },
    ];

    for (const { raw, expected } of invalidCases) {
        const result = MetadataUtil.normalizeTextSegments(raw);
        assert.equal(typeof result, "string");
        assert.match(result, new RegExp(expected.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    }
});
