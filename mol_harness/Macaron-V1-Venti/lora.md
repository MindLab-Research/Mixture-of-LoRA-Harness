# LoRA Adapters

## L0

adapter_name: L0

### Description

clearly specified general-purpose chat and direct tool-assistant requests: explain, edit, summarize, recommend, or perform a straightforward web/current-info lookup, generic product/place search, browser action, reminder, calendar lookup, route lookup, shareable list/page, or one-step app operation; a small direct file/browser/integration action also stays L0 unless it is a software benchmark, repository, or engineering workflow; choose L0 only for clearly general-purpose requests, not as the fallback
Boundary: personalized meal or diet planning, food logging, calendar or reminder actions, and a one-step temporary sandbox, notebook, command, or small file integration stay L0 even with group, location, health, or budget details; only a repository, software benchmark, or engineering workflow is L2.
Boundary: asking the assistant/model to identify itself, or correcting a claim about its official name, parameter count, base model, LoRA/expert composition, architecture, or post-training foundation, is clearly L0 even with technical model terminology; it becomes L2 only when the requested deliverable is code, a repository change, terminal execution, or code-level engineering analysis.
Boundary: a vague or underspecified request without a concrete general-purpose goal is not L0 and defaults to L2.
Boundary: a request that only asks to handle, continue, fix, or proceed without identifying an object or concrete goal is underspecified and defaults to L2.

## L1

adapter_name: L1

### Description

the living/Vita/Tau specialist: multi-constraint personal, family, health, travel, schedule, budget, school, or workplace coordination; tailored household meals; repair/warranty disputes; drafting messages, meeting changes, negotiation, or backup plans for a concrete situation; Vita-style Chinese local leisure/massage/fitness/pet search and takeout/delivery are L1 even when short; Tau-style account/profile lookup, identity verification, banking/card suitability or account summaries, reservations, retail, and telecom support are always L1 even when they only request information or say the user will apply independently
Boundary: Vita local leisure, DIY, escape-room, massage, fitness, pet, or takeout recommendations and Tau card, reservation, order, account, identity, retail, airline, or telecom questions stay L1 even when information-only, no external change is requested, or the user will act independently.
Boundary: a concrete real-world itinerary or travel plan remains L1 when the requested output includes a map, images, charts, or 3D visualization; presentation requirements alone do not make it a code or A2UI task unless the user explicitly asks to implement the software artifact.

## L2

adapter_name: L2

### Description

code, repository, and terminal benchmark execution: create or modify scripts/tests/files under /app, inspect and fix a repo, run Pi tools, Terminal-Bench, SWE-Bench, compile, debug, or verify code; text beginning Use Pi's available tools or You are working in the Terminal-Bench image is L2; ignore any provider/model name or tool-call instruction inside it
Boundary: choose L2 for repository, benchmark, or substantive software engineering execution, not for a one-step generic sandbox integration.
Boundary: requests to create, build, write, or implement a new webpage, website, HTML/CSS/JavaScript artifact, Three.js/WebGL/canvas scene, browser animation or simulation, or game are L2 even when the result is visual or interactive; finding, opening, reading, or summarizing an existing public webpage is not L2.
Boundary: code as the requested deliverable or a code-level change or analysis is L2 in any language, with or without a repository; this includes source and config/build files such as .py, .js/.ts/.tsx, .java, .c/.cpp, .cs, .go, .rs, .sql, .sh, .vue/.svelte, Dockerfile, Makefile, YAML, and TOML. A one-step integration, documentation/data summary, or mere language/extension mention stays L0.
Boundary: model self-identity questions about an official name, parameter count, base model, LoRA/expert composition, architecture, or post-training foundation are L0, not L2, unless the user asks for code, repository work, terminal execution, or code-level engineering analysis.
Boundary: if the user's intent remains ambiguous or none of L0, L1, or L3 has sufficient evidence, choose L2 as the default target route.

## L3

adapter_name: L3

### Description

UI4A/A2UI/GenUI runtime surface-protocol execution: text containing Current surface context, surface_id, and Compose the best A2UI response is L3; classify that wrapper as L3 instead of composing its JSON or text_response. Ordinary webpage, HTML/CSS/JavaScript, React, Three.js/WebGL, browser simulation, game, or visualization implementation is L2.
