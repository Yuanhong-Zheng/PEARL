ANNOTATE = """Annotate a sampled 64-second video window for learning runtime-defined visual concepts.
Use only visible pixels. Ignore subtitles, speech and movie knowledge. The question
will be asked at relative time 64. Frames have exact relative and source times.

Identify up to FOUR distinct clearly visible entities. Choose unambiguous
reference frames preferably in the first 16 seconds. Describe appearance and
where the target is IN THE REFERENCE IMAGE, never its real name or future actions.
Keep track of the same physical entity across views. Do not merge similar people.

Record up to 16 visually clear events with exact actor, action verb, object,
container, recipient/surface, location and supporting relative interval. Distinguish
placing a bag from pouring coins out of it; opening a door from walking through it;
holding an object from giving it away. A state such as sitting at the left of a
table is allowed, but specify its time and actor. Do not invent motions not visible
at this sampling rate. Prefer complete events after the reference frame.

Actively look for binding opportunities: two entities who interact with the same
object differently, hold different things, occupy different locations, or perform
different actions at the same time. Record BOTH actors' supported events, including
their actions at the end of the window. Named objects can also be concepts, but
prefer multiple actors who remain relevant in later frames.

Return JSON only:
{"usable":true,"reason":"short reason","concepts":[{"id":"C0","reference_frame":2,
"description":"the man on the left wearing a dark jacket"}],
"events":[{"id":"E0","actor":"C0","verb":"pours","object":"coins",
"container":"cloth bag","recipient":"counter","location":"a wooden counter",
"interval":[24,28],"visually_clear":true}],"scene":"short visual setting"}
If no entity and grounded event can be identified, return usable false with empty
concepts and events. Do not force useful annotations on title cards or unclear frames.
"""

GENERATE = """Create personalized visual question-answering data from the structured
entity/event annotation below. All questions concern the annotated scene, at
relative time 64. They will be checked independently against video pixels.

Return up to EIGHT grounded four-option questions with both past and current tasks.
Aim for four historical and four current questions when genuinely supported.
Past questions need evidence entirely before 56. Concept registration may be later
than historical evidence: only the question at 64 must follow all registrations.
Current evidence is in [56,64). Use the reference
frame ONLY to define the entity, not to supply the answer. Avoid questions whose
answers are already explicit in the appearance description or reference picture.
For past QA, prefer a transient earlier action, identity, or relation that is no
longer visible in the current clip. A stable object, outfit, seat, or location
that continues into [56,64) is unsuitable as a history-dependent question. If a
concept description already mentions holding an object, do not ask what it held.

Use {C0}, {C1}, etc. for user-defined concepts. Do not use real names or subtitles.
Also include identity/recipient questions whose ANSWER OPTIONS are runtime concepts,
when the visible events support them: e.g. "Who sits down in the current clip?"
with options {C0}, {C1}, {C2}, {C3}. This requires matching the visible actor to the
user's arbitrary concept name, even when the event itself is distinctive. With
fewer than four concepts, a distinct unregistered-person option can be used, or
ask for grounded sender/recipient relationships with four distinct concept-role
combinations. Use attribute relation or recipient. Do not invent extra registered
entities merely to fill options. Aim to mix target-conditioned attribute questions
and runtime-identity answers, rather than repeating one event as paraphrases.
Do not include numeric timestamps, frame/clip numbers, or answer-bearing hints in
the question. Anchor a past question to a distinctive visible event or scene so
that it is not an unrestricted ambiguous question about everything in the film.
The event verb must make its queried argument precise. All options must have the
same semantic role: e.g. contents poured out, or containers used; do not mix the
bag and the coins as competing answers to a vague question.

Current questions should explicitly say "now" or "in the current clip"; past
questions must clearly refer to earlier events. The attribute field must be one of:
object, container, action, location, order, recipient, relation.

Prefer hard distinctions: same person doing different things, similar actors with
different objects or positions, and events whose order matters. When two entities
have different answers to the SAME question, produce a swap pair: identical wording
except {C0}/{C1}, identical four options, different correct indices. Both members
use the same swap_pair_id. Paired variants are augmentations of one family.

PRIORITIZE shared-context pairs. Do not reveal a target through a unique action
presupposed by the question. If one actor throws a cloth and another removes it,
ask both "In the current clip, how does {TARGET} interact with the cloth?" using
the same action options. Do not ask "What does {TARGET} throw?", which reveals
the only throwing actor even with no concept definition. Neutral questions about
what a target is doing/holding or where it is at a SHARED moment should have
different plausible answers for different eligible entities. Do not put target
appearance clues or an actor's unique role into the question.

Avoid paraphrases of the same fact: vary event, target or queried attribute.
Use only facts in the annotation; evidence intervals must support every necessary
step and intersect no more than THREE 8-second clips. Correct answer positions
should vary. Do not choose None, All of the above, or multiple plausible answers.

Return JSON only with questions; concept definitions are fixed by the annotation
and need not be repeated. Example FORMAT ONLY, never copy unsupported events:
{"questions":[{"id":"q0","task":"current","concept_ids":["C0"],
"question":"In the current clip, how does {C0} interact with the cloth?",
"options":["Throws it","Removes it from their face","Folds it","Wipes a table"],
"answer_index":0,"evidence":[[58,59]],"event_ids":["E7"],"attribute":"action",
"swap_pair_id":"cloth_pair","negative_clip_indices":[]},
{"id":"q1","task":"current","concept_ids":["C1"],
"question":"In the current clip, how does {C1} interact with the cloth?",
"options":["Throws it","Removes it from their face","Folds it","Wipes a table"],
"answer_index":1,"evidence":[[60,63]],"event_ids":["E8"],"attribute":"action",
"swap_pair_id":"cloth_pair","negative_clip_indices":[]}]}
Give fewer questions rather than inventing evidence or ambiguous options.
"""

VERIFY = """Independently verify visual personalized QA. You are not given the proposed
correct answers; option order has been shuffled. Use the runtime entity references,
the sampled scene, and any older historical clips. The query is asked at the end
of the scene, relative time 64 (source time is provided). Older clips may contain
hard distractors OR other valid evidence; never assume they are negatives.

IMPORTANT: the CURRENT clip is ONLY relative [56,64). Relative [0,56) is
HISTORICAL, even though all its frames are shown for this audit. An event at
relative 23 seconds is historical, not current. Extra H clips are older still.

For each question, answer it, verify that the described reference really identifies
the target, that its evidence intervals suffice, that its event verb and answer
options have exactly one interpretation, and that it requires the concept binding.
Past questions should need history beyond the references and current clip. Current
questions should need the current pixels beyond the reference. Check small objects,
actor roles, object/container distinctions and event order carefully.

For confirmed_negative_labels, use ONLY L0..L6 (the local historical 8-second clips)
or H0..H5 (extra older clips shown below). A clip is negative only if the requested
event/actor/attribute is demonstrably absent or contradicted, not merely unlabeled.
An observed wrong actor, incompatible action, or different object can establish
a negative. If any shown clips clearly fail the requested predicate, explicitly
list at least one (prefer two) confirmed negatives. Judge the supplied sampled
pixels; do not require proof about unobserved moments between frames.
An L label identifies the ENTIRE explicitly labelled eight-second clip. Negative
means it does NOT contain supporting evidence. NEVER put any clip overlapping
the proposed evidence intervals into confirmed_negative_labels. If the proposed
interval is incorrect, flag evidence_sufficient false instead; listing a positive
as negative while declaring its evidence sufficient is a contradiction. An empty
negative list is preferable to an uncertain or contradictory negative label.
Do not mark an alternative valid occurrence as negative. If an older event makes
the question's answer non-unique, set unique_answer false.

Return JSON only:
{"questions":[{"id":"q0","answer_index":0,"concept_bindings_valid":true,
"visually_answerable":true,"unique_answer":true,"evidence_sufficient":true,
"requires_concept_binding":true,"no_reference_shortcut":true,
"requires_history_for_past":true,
"confirmed_negative_labels":["L0","H1"],"reason":"one short visual fact"}]}
Use false or null when unsure. Do not guess a label in order to pass an item.
"""

ABLATE = """Answer questions using ONLY the reduced visual input explicitly supplied.
You do NOT have the historical scene. Runtime references define arbitrary entity
names. For historical questions you additionally see the current final 8-second
clip; for current questions you only get references. Answer only when those pixels
actually determine the answer. Otherwise return answer_index null, even if a
plausible guess exists. Do not invent actor identities, events, or unseen context.
Answer indices refer to the displayed independently shuffled options.
Return JSON only: {"questions":[{"id":"q0","answer_index":null,
"sufficient_information":false,"reason":"missing earlier event"}]}.
"""

REFERENCE_ONLY = """You only see the runtime concept references below, not the queried
current or historical scene. Answer only if those references or their descriptions
actually determine a unique answer. Otherwise set answer_index null and
sufficient_information false. Do not use movie knowledge or guess unseen events.
Return JSON only: {"questions":[{"id":"q0","answer_index":null,
"sufficient_information":false,"reason":"missing event pixels"}]}.
"""

NO_HISTORY = """You see the runtime concept references and ONLY the CURRENT clip
at relative [56,64). The earlier historical scene is NOT available. Answer this
past question only if the references, descriptions, and current pixels determine
a unique answer. Otherwise return answer_index null and sufficient_information
false. Do not use movie knowledge or guess unseen events. The option order is
independently shuffled. Return JSON only:
{"questions":[{"id":"item","answer_index":null,
"sufficient_information":false,"reason":"missing historical evidence"}]}.
"""

NO_CONCEPT = """You see a video scene, but NO definitions of the arbitrary aliases in
the questions. Answer only if the visible context uniquely determines the answer
WITHOUT knowing which entity each alias identifies. If multiple entities would
give different answers and the mapping is missing, set answer_index null and
sufficient_information false. Do not assume aliases name entities in appearance
order. Do not use subtitles or outside knowledge.
Return JSON only: {"questions":[{"id":"q0","answer_index":null,
"sufficient_information":false,"reason":"missing alias binding"}]}.
"""
