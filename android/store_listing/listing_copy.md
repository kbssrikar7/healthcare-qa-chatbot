# MediQuery — Play Store Listing Copy (draft)

## Short description (max 80 characters)

```
Offline AI medical Q&A — grounded answers, sources cited, 100% on-device.
```
(74 characters)

## Full description (max 4000 characters)

```
MediQuery answers medical questions by searching a curated collection of
medical reference material and generating a grounded, source-cited answer —
entirely on your device, with no internet connection required and no data
ever leaving your phone.

HOW IT WORKS
Ask a question in plain language. MediQuery searches its on-device medical
knowledge base using a hybrid retrieval system (combining keyword and
semantic search), then generates an answer using an on-device language
model — grounded in the specific passages it found, which are shown to you
alongside every response so you can see exactly where the information came
from.

WHY ON-DEVICE MATTERS
- Your questions never leave your phone. There is no server, no account,
  and no network permission — the app is technically incapable of sending
  your data anywhere.
- Works without an internet connection, once installed.
- No ads, no analytics, no tracking.

WHAT IT'S FOR
MediQuery is an educational reference tool for general medical information —
symptoms, medications, and general health topics. It cites its sources so
you can evaluate the information yourself.

WHAT IT'S NOT
MediQuery is not a substitute for professional medical advice, diagnosis,
or treatment, and it is not a licensed medical device. Always consult a
qualified healthcare professional for any medical concern. In an emergency,
contact your local emergency services immediately.

ABOUT
MediQuery mobile is the on-device counterpart to the MediQuery web
application, built as part of an academic research project on
retrieval-augmented generation for healthcare question answering.
```
(1,340 characters — well under the 4000 limit; room to expand later)

## Required graphic assets (not yet produced — need actual design work)

- **App icon**: 512x512px, 32-bit PNG with alpha — have the source design
  (`public/mediquery-icon.svg`), needs export at this exact size.
- **Feature graphic**: 1024x500px JPG/PNG, no alpha — this is the banner
  shown at the top of the store listing. Not yet designed.
- **Phone screenshots**: minimum 2, recommended 4-8, JPG/PNG,
  16:9 or 9:16 aspect ratio, 320px-3840px per side. We already have real
  in-app screenshots from testing (e.g. the chat view with sources chips) —
  these could be used directly or lightly cleaned up (crop status bar,
  maybe add a device frame) rather than needing net-new screenshot design
  work.

## Content rating questionnaire — heads-up, not a blocker

Play Console will ask a standard questionnaire about content (violence,
sexual content, etc. — all "no" for this app) and about health-related
functionality specifically, given the app provides medical information.
Answer honestly and consistently with the privacy policy and description
above — this is where Play's Health Content policy scrutiny actually
happens, not at initial upload.
