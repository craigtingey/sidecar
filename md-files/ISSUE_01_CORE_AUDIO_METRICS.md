# Issue #1: Extract Core Audio Metrics from Training Sessions

**Repository:** `sidecar`  
**Labels:** `enhancement`, `audio-analysis`, `phase-1`, `priority-high`  
**Milestone:** Phase 1 - Audio Analysis Foundation  
**Dependencies:** None (can start immediately)  
**Estimated Effort:** 6-8 hours

---

## Overview

Enhance sidecar audio processing to extract 5 core performance metrics from training session audio. These metrics will be sent to WordPress for AI-enhanced coaching feedback.

## Background

Currently, the sidecar:
- Transcribes audio using OpenAI Whisper (with word-level timestamps)
- Performs speaker detection (USER vs BOT) using stereo channel energy analysis
- Sends formatted transcript to WordPress

This enhancement adds performance metric extraction to provide data-driven coaching insights.

---

## Acceptance Criteria

- [ ] **Speaking Pace (WPM)**: Calculate words per minute separately for USER and BOT
  - Use existing word timestamps from Whisper API
  - Formula: `(total_words / total_speaking_time_seconds) * 60`
  - Exclude silence/pauses from speaking time

- [ ] **Talk Ratio**: Calculate percentage of total speaking time per speaker
  - USER talk time vs BOT talk time
  - Formula: `(user_seconds / total_session_seconds) * 100`
  - Total should equal ~100% (excluding silence)

- [ ] **Filler Word Detection**: Count filler words for USER only
  - Target filler words: "um", "uh", "like", "you know", "sort of", "kind of"
  - Case-insensitive matching
  - Calculate rate: `filler_count / (total_user_time_minutes)`
  - Store both raw count and rate per minute

- [ ] **Average User Energy**: Calculate mean RMS energy for USER speech segments
  - Reuse existing RMS energy calculations from speaker detection
  - Average across all USER-attributed word segments
  - Normalize to 0-1 scale

- [ ] **Response Latency**: Measure average time between speakers (BOT finishes → USER starts)
  - **CRITICAL**: Account for artificial delays in Convai connection logic
  - **Only measure natural pauses**: Filter out system-induced delays
  - Look for delays > 200ms but < 3000ms (outside this range = likely system delay)
  - Calculate average of valid turn-taking gaps
  - Return in milliseconds

- [ ] **Enhanced Callback Payload**: Include metrics in WordPress callback
  - Add `audio_metrics` object to existing callback structure
  - Maintain backward compatibility (transcript still works standalone)
  - Handle missing/invalid audio gracefully (metrics = null)

- [ ] **Error Handling**: Graceful degradation if metric extraction fails
  - Log errors but don't block transcript callback
  - Return partial metrics if some calculations fail
  - Include `metrics_errors` array in callback if issues occurred

- [ ] **Unit Tests**: Add tests for metric extraction functions
  - Test WPM calculation with sample timestamps
  - Test filler word detection with known transcript
  - Test talk ratio calculation
  - Test response latency filtering (exclude artificial delays)

---

## Technical Implementation

### Example Callback Payload Structure

```json
{
  "session_id": "convai-abc-123",
  "scenario_id": 456,
  "transcript": "[USER]: Hello, I'm calling about your pest control services...\n[BOT]: Hi there! I'd be happy to help...",
  "server_token": "secret-token",
  "audio_metrics": {
    "speaking_pace_user_wpm": 165,
    "speaking_pace_bot_wpm": 145,
    "talk_ratio_user_pct": 48.3,
    "talk_ratio_bot_pct": 51.7,
    "total_user_time_sec": 145.2,
    "total_bot_time_sec": 155.8,
    "total_session_time_sec": 301.0,
    "filler_word_count": 8,
    "filler_words_per_minute": 2.1,
    "filler_words_detected": ["um", "uh", "like", "like", "you know", "um", "uh", "sort of"],
    "avg_user_energy_rms": 0.72,
    "avg_response_latency_ms": 450,
    "response_latency_samples": 12,
    "metrics_version": "1.0"
  }
}
```

### Response Latency - Artificial Delay Handling

**Problem:** Convai connection logic may introduce artificial delays that don't represent user thinking time.

**Solution:** Statistical filtering
```python
def calculate_response_latency(word_segments):
    """
    Calculate average response latency, filtering artificial delays.
    
    Only consider gaps where:
    - Previous speaker was BOT
    - Next speaker is USER
    - Gap duration is 200ms - 3000ms (natural human response range)
    
    Gaps outside this range are likely:
    - < 200ms: Overlap or immediate response (system artifact)
    - > 3000ms: Connection delay, user distraction, or system pause
    """
    latencies = []
    
    for i in range(len(word_segments) - 1):
        current = word_segments[i]
        next_word = word_segments[i + 1]
        
        # Only measure BOT → USER transitions
        if current['speaker'] == 'BOT' and next_word['speaker'] == 'USER':
            gap_ms = (next_word['start_time'] - current['end_time']) * 1000
            
            # Filter for natural response range
            if 200 <= gap_ms <= 3000:
                latencies.append(gap_ms)
    
    if latencies:
        return {
            'avg_response_latency_ms': round(sum(latencies) / len(latencies), 0),
            'response_latency_samples': len(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies)
        }
    
    return {
        'avg_response_latency_ms': None,
        'response_latency_samples': 0
    }
```

### Filler Word Detection

```python
FILLER_WORDS = [
    'um', 'uh', 'like', 'you know', 'sort of', 'kind of',
    'i mean', 'you see', 'actually', 'basically'
]

def detect_filler_words(transcript, speaker='USER'):
    """
    Count filler words in transcript for specified speaker.
    Returns both count and detected instances.
    """
    filler_count = 0
    detected = []
    
    # Extract only USER lines
    user_lines = [
        line.split(']: ', 1)[1] 
        for line in transcript.split('\n') 
        if line.startswith(f'[{speaker}]:')
    ]
    
    full_text = ' '.join(user_lines).lower()
    
    for filler in FILLER_WORDS:
        count = full_text.count(filler)
        filler_count += count
        detected.extend([filler] * count)
    
    return {
        'filler_word_count': filler_count,
        'filler_words_detected': detected
    }
```

---

## Implementation Notes

### Reuse Existing Infrastructure

- **Word timestamps**: Already available from Whisper `response_format: verbose_json`
- **Speaker detection**: Already implemented (stereo channel RMS energy)
- **RMS energy values**: Already calculated per word segment - just average for USER

### New Functions Needed

```python
# In server.py

def extract_audio_metrics(word_segments, transcript, audio_data=None):
    """
    Main function to extract all audio metrics.
    
    Args:
        word_segments: List of {word, start, end, speaker, energy}
        transcript: Formatted transcript string
        audio_data: Optional audio array for advanced analysis
    
    Returns:
        dict: Audio metrics or None if extraction fails
    """
    try:
        metrics = {}
        
        # Speaking pace
        pace = calculate_speaking_pace(word_segments)
        metrics.update(pace)
        
        # Talk ratio
        ratio = calculate_talk_ratio(word_segments)
        metrics.update(ratio)
        
        # Filler words
        fillers = detect_filler_words(transcript, 'USER')
        metrics.update(fillers)
        
        # User energy
        energy = calculate_avg_user_energy(word_segments)
        metrics.update(energy)
        
        # Response latency (with filtering)
        latency = calculate_response_latency(word_segments)
        metrics.update(latency)
        
        metrics['metrics_version'] = '1.0'
        
        return metrics
        
    except Exception as e:
        error_log(f'Failed to extract audio metrics: {e}')
        return None
```

### Error Handling

```python
# In callback logic
audio_metrics = extract_audio_metrics(word_segments, transcript)

callback_data = {
    'session_id': session_id,
    'transcript': transcript,
    'audio_metrics': audio_metrics  # Will be None if extraction failed
}

if audio_metrics is None:
    error_log('⚠️ Audio metrics extraction failed - sending transcript only')
```

---

## Testing Checklist

- [ ] Test with real training session audio (5+ minute recording)
- [ ] Verify WPM calculations against manual count
- [ ] Confirm filler word detection accuracy (>95%)
- [ ] Validate talk ratio sums to ~100%
- [ ] Test response latency filtering (exclude >3s gaps)
- [ ] Test error handling (malformed audio, missing timestamps)
- [ ] Test backward compatibility (callback works without metrics)
- [ ] Verify callback payload size (<5KB typical)

---

## Success Metrics

- ✅ Metrics extracted successfully on 100% of sessions with valid audio
- ✅ Callback payload includes all 5 core metrics
- ✅ WordPress receives and logs metrics (verified in Issue #2)
- ✅ Response latency excludes artificial delays (avg 200-1000ms typical)
- ✅ No performance degradation (transcription time increase <10%)

---

## Dependencies for Next Issues

**Issue #2** (WordPress) depends on this issue:
- Requires `audio_metrics` callback payload structure
- Should wait until this is deployed to dev environment
- Can develop in parallel using mock data

---

## Deployment Notes

1. Deploy to dev sidecar first
2. Test with 5-10 real training sessions
3. Verify metrics look reasonable (WPM 100-200, talk ratio 30-70%, etc.)
4. Monitor error logs for edge cases
5. Deploy to production after 48hr dev soak test

---

## Questions / Clarifications Needed

- [ ] Should we exclude BOT energy from metrics? (Currently USER only)
- [ ] What's the expected range for "good" response latency? (Currently 200-1000ms)
- [ ] Should we track interruptions (USER starts before BOT finishes)?

---

**Created:** December 2, 2025  
**Priority:** HIGH - Blocks Phase 1 completion  
**Start After:** Immediate (no blockers)
