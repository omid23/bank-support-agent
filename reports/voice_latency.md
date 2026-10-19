# Voice turn latency (offline)

End of caller speech → first sentence handed to TTS, through `AgentProcessor` with the real agent runtime (knowledge search, guard). The model is a stand-in with a fixed delay per call (2 calls per turn: search, then answer). STT, TTS and network are not included.

| model ms / call | turns | first output p50 | p95 | agent turn p50 | p95 | search p50 | p95 | fillers |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 97 | 279.5 | 294.7 | 28.4 | 43.2 | 27.3 | 41.6 | 0 |
| 400 | 97 | 1052.3 | 1054.0 | 831.9 | 842.2 | 30.0 | 39.8 | 97 |

End-of-turn grace: 250 ms; filler after 800 ms (the filler is the first output when the agent is slower than that).
