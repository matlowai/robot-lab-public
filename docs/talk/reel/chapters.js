window.REEL_CHAPTERS = {
  "duration": 92.888,
  "chapters": [
    {
      "t": 0,
      "label": "Cold open"
    },
    {
      "t": 4.5,
      "label": "Race"
    },
    {
      "t": 19.593,
      "label": "Scale"
    },
    {
      "t": 25.693,
      "label": "Reward"
    },
    {
      "t": 32.198,
      "label": "Training"
    },
    {
      "t": 38.703,
      "label": "Denominator"
    },
    {
      "t": 44.828,
      "label": "Real or fake"
    },
    {
      "t": 50.433,
      "label": "No release"
    },
    {
      "t": 56.923,
      "label": "Let go"
    },
    {
      "t": 63.413,
      "label": "RL gate"
    },
    {
      "t": 70.103,
      "label": "Checker"
    },
    {
      "t": 76.593,
      "label": "Fine-tune"
    },
    {
      "t": 82.883,
      "label": "Imagination"
    },
    {
      "t": 88.288,
      "label": "Close"
    }
  ],
  "stops": [
    {
      "t": 19.093,
      "title": "The test was timed to the slow robot",
      "why": "The walkers followed a timetable fitted to the no-avoidance robot, so the two faster robots passed every encounter before its walker arrived and never faced the test; the only contacts were late walkers reaching the charger after they parked. The fix: pedestrians that react to whichever robot is in front of them."
    },
    {
      "t": 31.073,
      "title": "Whatever you don't price in, it ignores",
      "why": "Every weight on this receipt, from +1 per metre of progress to −10 for contact, is a design choice, and the policy optimizes exactly what is on it."
    },
    {
      "t": 37.578,
      "title": "Same algorithm, harder world",
      "why": "With straight-line walkers PPO reached 96% training success; once pedestrians reacted to the robot, the same setup ended between 31% and 41%."
    },
    {
      "t": 43.453,
      "title": "Report the denominator",
      "why": "On hits per commit the RL policy looked tied with the planner (37.2% vs 38.5%), but it bumped ordinary people in 61 of 80 runs versus 27 and took longer (462 s vs 377 s median)."
    },
    {
      "t": 49.308,
      "title": "Real or fake?",
      "why": "All seven successes the detector reported overnight were fake; with the jaw pre-opened, one of six attempts was a genuine grasp, lift, carry and place."
    },
    {
      "t": 55.573,
      "title": "It never saw itself let go",
      "why": "In 48 of the 59 night-1 attempts where it lifted the object but did not place it, the policy never commanded the gripper open; the demo recorder had reset the scene the instant the object landed, so a typical demo held one release frame in about 145."
    },
    {
      "t": 62.163,
      "title": "Fix the data, not the model",
      "why": "Night 2 kept recording through release and retreat (a median of 42 release frames in each of 862 demos) and weighted gripper open/close frames 5× in the loss: 73 of 144 strict real places, 12 of 48 on objects it never trained on."
    },
    {
      "t": 68.753,
      "title": "A learned release gate",
      "why": "On sealed test seeds never used in training, with identical layouts for both, the night-1 model alone made 58 of 465 strict places on training objects and 22 of 236 on objects it never trained on; with the residual RL gate on top, 382 of 465 and 122 of 236."
    },
    {
      "t": 75.143,
      "title": "Check it after it settles",
      "why": "On 95 end-of-attempt frames scored against the simulator’s ground truth, Gemma 4 12B was right 94.7% of the time (5 false “done” in 50 failures) and Qwen 3.8 27B 97.9% (1 in 50), with the real places photographed after the arm had released, retreated and the scene had settled."
    },
    {
      "t": 81.233,
      "title": "One error traded for another",
      "why": "A 4-bit QLoRA fine-tune of Gemma cut false “done” answers on settled frames from 12 to 3 of 50, but it now misses 41 of 45 real places photographed at the instant of release (base: 15) and 9 of 12 on objects it never trained on (base: 3), so it is not deployable yet."
    },
    {
      "t": 85.363,
      "title": "It imagines the next two seconds",
      "why": "Alongside its next moves the model predicts video of what it expects to see; that video is usually thrown away, but it shows what the model thinks will happen."
    }
  ]
};
