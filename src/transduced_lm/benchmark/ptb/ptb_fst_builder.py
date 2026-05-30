"""
PTB Tokenizer FST built using pynini.

This uses pynini's cdrewrite rules to create a compact FST that matches
NLTK's TreebankWordTokenizer behavior. The pynini FST is then converted to the native
FST format for use with the Rust decomposition backend.

Period handling uses pynini's built-in [EOS] boundary symbol in cdrewrite rules
to match NLTK's TreebankWordTokenizer behavior (periods only separated at end of string).

Based on the reference implementation in ptb.py.
"""
import pynini
from pynini import cdrewrite, cross, union

from .fst import FST
from .fsa import EPSILON as NATIVE_EPSILON


EPS = "<eps>"
MARKER = 257    # Out-of-band word boundary marker (not a real byte) Removed at the end
SEP = 258       # Output separator symbol (word boundary in final output)

# String forms for pynini symbol table (pynini requires string names)
_MARKER_STR = str(MARKER)
_SEP_STR = str(SEP)






def _char_to_byte(ch, symbols):
    """Convert a single ASCII character to pynini acceptor."""
    return pynini.accep(str(ord(ch)), token_type=symbols)


def _chars_to_bytes(s, symbols):
    """Convert a string to concatenation of byte acceptors."""
    result = pynini.accep("", token_type=symbols)
    for ch in s:
        result = result + pynini.accep(str(ord(ch)), token_type=symbols)
    return result


def _build_separator_inserter(symbols, ext_symbols):
    """
    Build an FST that converts MARKER bytes to SEP and collapses consecutive markers.
    """
    fst = pynini.Fst()
    start = fst.add_state()
    in_markers = fst.add_state()
    fst.set_start(start)
    fst.set_final(start)
    fst.set_final(in_markers)

    marker_id = ext_symbols.find(_MARKER_STR)
    sep_id = ext_symbols.find(_SEP_STR)

    # From start: first marker -> output SEP, go to in_markers
    fst.add_arc(start, pynini.Arc(marker_id, sep_id, 0, in_markers))

    # From in_markers: more markers -> consume without output
    fst.add_arc(in_markers, pynini.Arc(marker_id, 0, 0, in_markers))

    # From both states: non-marker -> pass through
    for idx in range(1, symbols.num_symbols()):
        sym = symbols.find(idx)
        if sym is not None and idx != marker_id:
            fst.add_arc(start, pynini.Arc(idx, idx, 0, start))
            fst.add_arc(in_markers, pynini.Arc(idx, idx, 0, start))

    fst.set_input_symbols(ext_symbols)
    fst.set_output_symbols(ext_symbols)
    return fst.optimize()


def build_ptb_fst_pynini():
    """
    Build PTB tokenizer FST using pynini, then convert to native format.

    Returns a native FST with ~130 states that implements Penn Treebank tokenization.
    """
    # Build symbol table: bytes 0-255 + out-of-band MARKER
    symbols = pynini.SymbolTable()
    symbols.add_symbol(EPS, 0)
    for bt in range(256):
        symbols.add_symbol(str(bt), bt + 1)
    symbols.add_symbol(_MARKER_STR, 257)

    # Extended symbols: adds SEP for output
    ext_symbols = pynini.SymbolTable()
    ext_symbols.add_symbol(EPS, 0)
    for bt in range(256):
        ext_symbols.add_symbol(str(bt), bt + 1)
    ext_symbols.add_symbol(_MARKER_STR, 257)
    ext_symbols.add_symbol(_SEP_STR, 258)

    # Helper function shortcuts
    def cb(ch):
        return _char_to_byte(ch, symbols)

    def cs(s):
        return _chars_to_bytes(s, symbols)

    # Common acceptors
    MARKER_ACC = pynini.accep(_MARKER_STR, token_type=symbols)
    SPACE = cb(" ")
    # All single-byte characters that Python str.split() treats as whitespace
    WHITESPACE = union(
        cb(" "), cb("\t"), cb("\n"), cb("\r"), cb("\v"), cb("\f"),
        cb("\x1c"), cb("\x1d"), cb("\x1e"), cb("\x1f"),
    )
    APOS = cb("'")
    QUOTE = cb('"')
    DOT = cb(".")
    DASH = cb("-")
    BACKTICK = cb("`")
    DOUBLE_BACKTICK = BACKTICK + BACKTICK

    # Build sigma (all bytes 0-255 + MARKER)
    sigma = pynini.union(
        *[pynini.accep(str(i), token_type=symbols) for i in range(256)],
        pynini.accep(_MARKER_STR, token_type=symbols),
    )
    sigma_star = pynini.closure(sigma)

    # Identity FST as base (passes through all bytes)
    identity_fst = pynini.Fst()
    s = identity_fst.add_state()
    identity_fst.set_start(s)
    identity_fst.set_final(s)
    for idx in range(symbols.num_symbols()):
        if idx != 0:
            identity_fst.add_arc(s, pynini.Arc(idx, idx, 0, s))
    identity_fst.set_input_symbols(symbols)
    identity_fst.set_output_symbols(symbols)
    identity_fst = identity_fst.closure()

    # === Quote handling ===
    # Matches NLTK's STARTING_QUOTES then ENDING_QUOTES order:
    # 1. '' after space/brackets -> `` (STARTING_QUOTES rule 3 for '{2})
    # 2. Remaining '' -> separated '' (ENDING_QUOTES rule 1)
    # 3. " after word chars -> '' (closing double-quote)
    # 4. " after space/brackets -> `` (opening double-quote)
    # 5. Remaining " -> `` (e.g., at start of string)
    # 6. Wrap existing `` with markers

    # Characters that can precede a CLOSING quote
    # Include UTF-8 continuation bytes (128-191 / 0x80-0xBF) for accented characters
    UTF8_CONTINUATION_BYTES = union(*[pynini.accep(str(i), token_type=symbols) for i in range(128, 192)])
    WORD_CHARS = union(
        *[cb(chr(i)) for i in range(ord('a'), ord('z')+1)],  # lowercase
        *[cb(chr(i)) for i in range(ord('A'), ord('Z')+1)],  # uppercase
        *[cb(chr(i)) for i in range(ord('0'), ord('9')+1)],  # digits
        APOS, cb("!"), cb("?"), cb("."), cb(")"), cb(","),  # punctuation
        cb(";"), cb(":"),  # semicolon and colon can precede closing quote
        cb("]"),  # closing bracket can precede closing quote
        UTF8_CONTINUATION_BYTES,  # UTF-8 continuation bytes for accented chars
    )

    # Characters that can precede an OPENING quote
    opening_context = union(cb("("), cb("["), cb("{"), cb("<"), SPACE)
    DOUBLE_APOS = APOS + APOS

    # Rule 1: '' after space/brackets -> `` (opening double-single-quotes)
    # NLTK STARTING_QUOTES rule 3: ([ \(\[{<])(\"{2}|\'{2}) -> \1 ``
    opening_double_apos_rule = cdrewrite(
        cross(DOUBLE_APOS, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        opening_context.plus,
        "", sigma_star
    )

    # Rule 2: Remaining '' -> MARKER '' MARKER (closing/separating)
    # NLTK ENDING_QUOTES rule 1: '' -> ' '' '
    closing_double_apos_rule = cdrewrite(
        cross(DOUBLE_APOS, MARKER_ACC + APOS + APOS + MARKER_ACC),
        "", "", sigma_star
    )

    # Rule 3: " after word chars -> '' (closing quote) - must run FIRST among "-rules
    closing_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + APOS + APOS + MARKER_ACC),
        WORD_CHARS,
        "", sigma_star
    )

    # Rule 4: " after space/brackets -> `` (opening quote)
    opening_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        opening_context.plus,
        "", sigma_star
    )

    # Rule 5a: "" (two quotes) at beginning of string -> `` ``
    # In NLTK, ^" converts the first, then the inserted space lets rule 3 convert
    # the second.  Must come before the single-" BOS rule to match greedily.
    bos_double_quote_rule = cdrewrite(
        cross(QUOTE + QUOTE,
              MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC
              + MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        "[BOS]", "", sigma_star
    )

    # Rule 5b: single " at beginning of string -> `` (NLTK STARTING_QUOTES rule 1: ^")
    bos_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        "[BOS]", "", sigma_star
    )

    # Rule 5c: Remaining " -> '' (NLTK ENDING_QUOTES rule 2: all other " become '')
    remaining_quote_rule = cdrewrite(
        cross(QUOTE, MARKER_ACC + APOS + APOS + MARKER_ACC),
        "", "", sigma_star
    )

    # Rule 6: `` -> MARKER `` MARKER (for manually entered backticks)
    backtick_rule = cdrewrite(
        cross(DOUBLE_BACKTICK, MARKER_ACC + DOUBLE_BACKTICK + MARKER_ACC),
        "", "", sigma_star
    )

    # Compose: '' rules first, then " rules, then backtick wrapping
    # bos_double_quote_rule before bos_quote_rule so "" at BOS is handled greedily
    quotes_fst = (opening_double_apos_rule @ closing_double_apos_rule
                  @ closing_quote_rule @ opening_quote_rule
                  @ bos_double_quote_rule @ bos_quote_rule
                  @ remaining_quote_rule
                  @ backtick_rule).optimize()

    # === Punctuation ===
    DIGIT = union(*[cb(str(i)) for i in range(10)])
    NON_DIGIT = pynini.difference(sigma, DIGIT).optimize()

    # , and : before non-digit
    # NLTK regex: ([:,])([^\d]) - only separate when followed by non-digit
    punct_1 = cdrewrite(
        cross(cb(","), MARKER_ACC + cb(",") + MARKER_ACC) |
        cross(cb(":"), MARKER_ACC + cb(":") + MARKER_ACC),
        "", NON_DIGIT, sigma_star
    )

    # , and : at end of string only (before EOS)
    # NLTK regex: ([:,])$ - only at end of string
    punct_2 = cdrewrite(
        cross(cb(","), MARKER_ACC + cb(",") + MARKER_ACC) |
        cross(cb(":"), MARKER_ACC + cb(":") + MARKER_ACC),
        "", "[EOS]", sigma_star
    )

    # Ellipsis
    ellipsis_rule = cdrewrite(
        cross(DOT + DOT + DOT, MARKER_ACC + DOT + DOT + DOT + MARKER_ACC),
        "", "", sigma_star
    )

    # Special punct: ; @ # % & $
    special_punct = [cb(c) for c in ";@#%&$"]
    punct_4 = cdrewrite(
        union(*[cross(sym, MARKER_ACC + sym + MARKER_ACC) for sym in special_punct]),
        "", "", sigma_star
    )

    # Period at end - ONLY separate when followed by EOS (end of string)
    # This matches NLTK's behavior which uses $ regex anchor
    # NLTK regex: ([^\.])(\.)([\]\)}>"']*)\s*$
    # Period is separated when followed by optional closing punct, optional whitespace, then EOS
    NON_DOT = pynini.difference(sigma, DOT).optimize()

    # Optional closing punctuation that can appear after period before EOS
    # Includes MARKER_ACC since quotes are wrapped with markers at this point
    CLOSING_PUNCT = union(
        cb("]"), cb(")"), cb("}"), cb(">"),
        APOS, MARKER_ACC,
    )
    CLOSING_PUNCT_STAR = pynini.closure(CLOSING_PUNCT)
    WHITESPACE_STAR = pynini.closure(WHITESPACE)

    punct_5 = cdrewrite(
        cross(DOT, MARKER_ACC + DOT),
        NON_DOT,
        CLOSING_PUNCT_STAR + WHITESPACE_STAR + pynini.accep("[EOS]"),
        sigma_star
    )

    # ? and !
    punct_6 = cdrewrite(
        cross(cb("?"), MARKER_ACC + cb("?") + MARKER_ACC) |
        cross(cb("!"), MARKER_ACC + cb("!") + MARKER_ACC),
        "", "", sigma_star
    )

    punct_fst = (punct_1 @ punct_2 @ ellipsis_rule @ punct_4 @ punct_5 @ punct_6).optimize()

    # === Brackets and parens ===
    parens_chars = [cb(c) for c in "[](){}><"]
    parens_brackets_fst = cdrewrite(
        union(*[cross(sym, MARKER_ACC + sym + MARKER_ACC) for sym in parens_chars]),
        "", "", sigma_star
    ).optimize()

    # === Double dashes ===
    double_dashes_fst = cdrewrite(
        cross(DASH + DASH, MARKER_ACC + DASH + DASH + MARKER_ACC),
        "", "", sigma_star
    ).optimize()

    # === Clitics and contractions ===
    # NLTK's clitic rules use [^' ] (literal space) as left context and
    # literal space as right context.  Contractions use \b (word boundary)
    # which matches at any whitespace.  We need separate context sets.
    NON_APOS_OR_SPACE_OR_MARKER = pynini.difference(sigma, union(APOS, SPACE, MARKER_ACC)).optimize()
    # Clitic right context: literal space, marker, apostrophe, or EOS
    # (NLTK pattern: ([^' ])('s|...) ' ' — trailing literal space)
    CLITIC_SEP = union(MARKER_ACC, SPACE, APOS, "[EOS]")

    # Clitics: 's 'm 'd
    clitics_1 = [cs(c) for c in ["'s", "'m", "'d", "'S", "'M", "'D"]]
    endq_3 = cdrewrite(
        union(*[cross(clit, MARKER_ACC + clit) for clit in clitics_1]),
        NON_APOS_OR_SPACE_OR_MARKER, CLITIC_SEP, sigma_star
    )

    # Double clitics and n't - must come before apos_rule
    clitics_2 = [
        cs("'ll"), cs("'LL"),
        cs("'re"), cs("'RE"),
        cs("'ve"), cs("'VE"),
        cs("n't"), cs("N'T"),
    ]
    endq_4 = cdrewrite(
        union(*[cross(clit, MARKER_ACC + clit) for clit in clitics_2]),
        NON_APOS_OR_SPACE_OR_MARKER, CLITIC_SEP, sigma_star
    )

    # Standalone apostrophe at word end (before space or EOS)
    # NLTK rule: ([^' ])(' ) - separates trailing ' before literal space
    CLITIC_SEP_OR_EOS = union(MARKER_ACC, SPACE, "[EOS]")
    apos_rule = cdrewrite(
        cross(APOS, MARKER_ACC + APOS),
        NON_APOS_OR_SPACE_OR_MARKER, CLITIC_SEP_OR_EOS, sigma_star
    )

    clitics_fst = (endq_3 @ endq_4 @ apos_rule).optimize()

    # === Contractions ===
    # NLTK uses \b (word boundary) which matches at any whitespace
    SEP_OR_BOS = union(MARKER_ACC, WHITESPACE, "[BOS]")
    CONTRACTION_SEP = union(MARKER_ACC, WHITESPACE, APOS, "[EOS]")

    # NLTK uses (?i) so all contractions are case-insensitive.
    # We enumerate lowercase, Title, and UPPER; mixed-case is rare enough to skip.
    contractions_base = [
        ("cannot", "can", "not"),
        ("gonna", "gon", "na"),
        ("gotta", "got", "ta"),
        ("lemme", "lem", "me"),
        ("wanna", "wan", "na"),
        ("gimme", "gim", "me"),
        ("d'ye", "d", "'ye"),
        ("more'n", "more", "'n"),
    ]
    contractions_raw = []
    for orig, p1, p2 in contractions_base:
        contractions_raw.append((orig, p1, p2))                          # lowercase
        contractions_raw.append((orig.capitalize(), p1.capitalize(), p2)) # Title
        contractions_raw.append((orig.upper(), p1.upper(), p2.upper()))   # UPPER

    contractions_fsts = []
    for orig, part1, part2 in contractions_raw:
        rule = cdrewrite(
            cross(cs(orig), MARKER_ACC + cs(part1) + MARKER_ACC + MARKER_ACC + cs(part2) + MARKER_ACC),
            SEP_OR_BOS, CONTRACTION_SEP, sigma_star
        )
        contractions_fsts.append(rule)

    contractions_fst = contractions_fsts[0]
    for c in contractions_fsts[1:]:
        contractions_fst = contractions_fst @ c
    contractions_fst = contractions_fst.optimize()

    # === CONTRACTIONS3: 'tis, 'twas ===
    # NLTK regex: ('t)(is)\b and ('t)(was)\b — preceded by space (text is padded)
    contractions3_raw = [("'tis", "'t", "is"), ("'twas", "'t", "was"),
                         ("'Tis", "'T", "is"), ("'Twas", "'T", "was"),
                         ("'TIS", "'T", "IS"), ("'TWAS", "'T", "WAS")]
    contractions3_fsts = []
    for orig, p1, p2 in contractions3_raw:
        rule = cdrewrite(
            cross(cs(orig), MARKER_ACC + cs(p1) + MARKER_ACC + MARKER_ACC + cs(p2) + MARKER_ACC),
            SEP_OR_BOS, CONTRACTION_SEP, sigma_star
        )
        contractions3_fsts.append(rule)
    contractions3_fst = contractions3_fsts[0]
    for c in contractions3_fsts[1:]:
        contractions3_fst = contractions3_fst @ c
    contractions3_fst = contractions3_fst.optimize()

    # === Whitespace to marker ===
    # Convert all whitespace to markers (matching NLTK's str.split() behavior)
    space_to_marker = cdrewrite(
        cross(WHITESPACE, MARKER_ACC),
        "", "", sigma_star
    ).optimize()

    # === Compose all rules ===
    core_fst = identity_fst
    core_fst = (core_fst @ quotes_fst).optimize()  # Handle all quote types
    core_fst = (core_fst @ punct_fst).optimize()
    core_fst = (core_fst @ parens_brackets_fst).optimize()
    core_fst = (core_fst @ double_dashes_fst).optimize()
    core_fst = (core_fst @ clitics_fst).optimize()  # Handle clitics ('s, 'll, n't, etc.)
    core_fst = (core_fst @ contractions_fst).optimize()
    core_fst = (core_fst @ contractions3_fst).optimize()
    core_fst = (core_fst @ space_to_marker).optimize()

    # === Separator inserter ===
    core_fst.set_input_symbols(ext_symbols)
    core_fst.set_output_symbols(ext_symbols)

    sep_fst = _build_separator_inserter(symbols, ext_symbols)

    final_fst = (core_fst @ sep_fst).optimize()

    # === Convert to native FST ===
    native_fst = FST()

    for state in final_fst.states():
        native_fst.states.add(state)

    native_fst.add_start(final_fst.start())

    for state in final_fst.states():
        final_weight = final_fst.final(state)
        if final_weight != pynini.Weight.zero(final_fst.weight_type()):
            native_fst.add_stop(state)

    marker_id = ext_symbols.find(_MARKER_STR)
    for state in final_fst.states():
        for arc in final_fst.arcs(state):
            # Skip arcs with MARKER as input (unreachable — input never contains MARKER)
            if arc.ilabel == marker_id:
                continue
            input_sym = NATIVE_EPSILON if arc.ilabel == 0 else int(ext_symbols.find(arc.ilabel))
            output_sym = NATIVE_EPSILON if arc.olabel in (0, marker_id) else int(ext_symbols.find(arc.olabel))
            native_fst.add_arc(state, input_sym, output_sym, arc.nextstate)

    return native_fst
