# BlackJack Bot

A humanoid robot that plays BlackJack against a human player, detects cards via computer vision, calculates probabilities, makes Hit/Stand decisions, and verbally explains its reasoning in real time.

## Team
- Or Smolarchik
- Eyal Abisdris
- Lecturer: Idan Tobis
- Course: Software Development for Human & Robot Humanoid Interaction
- Proposal submitted: 17/05/26
- Proposal document: `C:\Users\eyala\Desktop\robotics modules\BlackJack_Bot_Proposal.docx`

## Project Goals
1. Real-time playing card recognition from a camera feed.
2. BlackJack decision-making engine (Hit / Stand).
3. Basic card-counting mechanism.
4. Explainable AI: the robot verbally justifies each decision.
5. Social interaction: speech, body gestures, emotional reactions to wins/losses.
6. Study how explanations and social behavior affect user trust and engagement.

## System Architecture
Four modules, pipelined:

1. **Computer Vision Module** - fixed overhead camera detects cards on the table.
2. **Decision-Making Module** - computes hand value, probabilities, basic card counting, outputs Hit or Stand.
3. **Explainable AI Module** - generates a natural-language justification for the decision.
4. **Interaction Module** - speech (TTS), body movements, win/loss animations on the robot.

Example explanation the robot should produce:
> "I currently have 16 against the dealer's 10. Any card above 5 will cause me to lose, and many low cards have already left the deck, increasing the probability of drawing a high card. Therefore, I will stand."

## Tech Stack
- **Robot:** Darwin Mini (16 XL-320 Dynamixel servos, OpenCM9.04 controller)
- **Robot control from Python:** pypot library, likely via USB2AX or U2D2 adapter to bypass on-board controller
- **Camera:** Webcam or smartphone camera (fixed overhead)
- **Language:** Python
- **CV:** OpenCV; start with template matching, upgrade to YOLO if needed
- **Speech:** Text-to-Speech library
- **Hardware:** Laptop, deck of cards

## Open Questions for Lecturer
- Does the lab already have a USB2AX or U2D2 adapter for Python control of the Darwin Mini?
- Is pypot or any existing Python setup already in place?
- If not, can we use the OpenCM IDE / RoboPlus Task as a fallback for robot motion, while running CV + decisions in Python on the laptop?

## Current Status
Proposal stage. No code written yet. First implementation target: card recognition prototype on laptop (template matching) before integrating with the robot.

## Planned Folder Layout
```
BlackJackBot/
  vision/          # card detection
  decision/        # BlackJack logic, probabilities, card counting
  explain/         # explanation generation
  interaction/     # robot motion + TTS
  data/            # captured card images / templates
  tests/
```

## Known Risks
- Card recognition fragile under poor or uneven lighting.
- Latency between detection, decision, and robot response.
- Synchronization between speech and body movement.
- Darwin Mini hardware limitations (motion range, processing).
- Python <-> Darwin Mini integration may need extra hardware (adapter).

## Ethical Notes
- No sensitive personal data collected.
- Research/demonstration use only.
- Participants will be briefed on the system's purpose before any user study.
