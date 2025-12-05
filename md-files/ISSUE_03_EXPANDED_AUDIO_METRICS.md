# Issue #3: Expand Audio Metrics for Scoreboard Features

**Repository:** `sidecar`  
**Labels:** `enhancement`, `audio-analysis`, `phase-2`  
**Milestone:** Phase 2 - Performance Scoreboard  
**Dependencies:** Issue #6 (Core metrics foundation)  
**Estimated Effort:** 8-10 hours

---

## Overview

Expand audio analysis to include advanced metrics needed for performance scoring and trending features. These metrics enable detailed performance scorecards and identify specific coaching opportunities.

## Background

Phase 1 (Issue #6) established 5 core metrics:
- Speaking pace (WPM)
- Talk ratio (%)
- Filler word count/rate
- User energy (RMS)
- Response latency (ms)

Phase 2 adds **7 advanced metrics** for deeper performance analysis:
- Turn-taking patterns (monologue detection)
- Question analysis (open vs closed)
- Pause/silence patterns
- Pitch variation (prosody/tone)

These metrics enable the "scoreboard" UI and trending/analytics features.

---

## Acceptance Criteria

### 1. Turn-Taking Metrics

- [ ] **User Turn Count**: Total number of times USER speaks
- [ ] **Average Turn Length**: Mean duration of USER speaking segments (seconds)
- [ ] **Max Turn Length**: Longest single USER speaking segment (seconds)
- [ ] **Long Monologue Detection**: Count of USER turns >30 seconds
  - Flag potential issue: rep dominating conversation
  - Common in "pitch mode" vs consultative selling

**Technical Implementation:**
```python
def calculate_turn_metrics(word_segments):
    """
    Analyze turn-taking patterns from word segments.
    
    A "turn" is a continuous segment where the same speaker talks.
    Segments are separated when speaker changes.
    """
    turns = []
    current_turn = None
    
    for segment in word_segments:
        if segment['speaker'] == 'USER':
            if current_turn is None:
                # Start new turn
                current_turn = {
                    'start': segment['start_time'],
                    'end': segment['end_time']
                }
            else:
                # Extend current turn
                current_turn['end'] = segment['end_time']
        else:
            # Speaker changed to BOT
            if current_turn:
                turns.append(current_turn)
                current_turn = None
    
    # Don't forget last turn
    if current_turn:
        turns.append(current_turn)
    
    turn_lengths = [(t['end'] - t['start']) for t in turns]
    
    return {
        'user_turn_count': len(turns),
        'avg_turn_length_sec': round(sum(turn_lengths) / len(turn_lengths), 1) if turns else 0,
        'max_turn_length_sec': round(max(turn_lengths), 1) if turns else 0,
        'long_monologues_count': sum(1 for length in turn_lengths if length > 30)
    }
```

### 2. Question Analysis

- [ ] **Total Question Count**: Number of questions asked by USER
- [ ] **Open Question Count**: Questions starting with who/what/when/where/why/how
- [ ] **Closed Question Count**: Questions starting with do/is/can/are/did/will/should
- [ ] **Question Ratio**: Percentage of questions that are open-ended

**Detection Logic:**
```python
OPEN_QUESTION_PATTERNS = [
    r'\b(who|what|when|where|why|how)\b',
    r'\btell me (about|more)\b',
    r'\bwalk me through\b',
    r'\bhelp me understand\b'
]

CLOSED_QUESTION_PATTERNS = [
    r'\b(do|does|did|is|are|was|were|can|could|would|will|should|have|has)\b.*\?',
    r'\bright\?$',
    r'\bcorrect\?$'
]

def analyze_questions(transcript, speaker='USER'):
    """
    Count and classify questions in transcript.
    
    Uses regex patterns to detect question types.
    Looks for question marks and leading patterns.
    """
    # Extract USER lines
    user_lines = [
        line.split(']: ', 1)[1] 
        for line in transcript.split('\n') 
        if line.startswith(f'[{speaker}]:')
    ]
    
    questions = [line for line in user_lines if '?' in line]
    
    open_questions = []
    closed_questions = []
    
    for q in questions:
        q_lower = q.lower()
        
        # Check for open question patterns
        is_open = any(re.search(pattern, q_lower) for pattern in OPEN_QUESTION_PATTERNS)
        
        # Check for closed question patterns
        is_closed = any(re.search(pattern, q_lower) for pattern in CLOSED_QUESTION_PATTERNS)
        
        if is_open and not is_closed:
            open_questions.append(q)
        elif is_closed and not is_open:
            closed_questions.append(q)
        else:
            # Ambiguous - default to closed if starts with verb, else open
            if re.match(r'^\s*(do|is|can|are|did|will|should)', q_lower):
                closed_questions.append(q)
            else:
                open_questions.append(q)
    
    total = len(questions)
    open_count = len(open_questions)
    closed_count = len(closed_questions)
    
    return {
        'question_count_total': total,
        'question_count_open': open_count,
        'question_count_closed': closed_count,
        'question_ratio_open_pct': round((open_count / total * 100), 1) if total > 0 else 0
    }
```

### 3. Pause & Silence Metrics

- [ ] **Total Silence Percentage**: % of session time with no speech
- [ ] **Long Pause Count**: Number of silent gaps >2 seconds (USER is silent after BOT speaks)
- [ ] **Average Pause Duration**: Mean length of silent gaps (seconds)

**Technical Notes:**
```python
def calculate_silence_metrics(word_segments, total_session_time):
    """
    Analyze silence/pause patterns.
    
    Silence = gaps between word segments.
    Long pause = >2s gap where USER is expected to respond.
    """
    silences = []
    long_pauses = []
    
    for i in range(len(word_segments) - 1):
        current = word_segments[i]
        next_word = word_segments[i + 1]
        
        gap = next_word['start_time'] - current['end_time']
        
        if gap > 0.1:  # Ignore tiny gaps (<100ms)
            silences.append(gap)
            
            # Long pause: BOT finishes, USER takes >2s to respond
            if current['speaker'] == 'BOT' and next_word['speaker'] == 'USER' and gap > 2.0:
                long_pauses.append(gap)
    
    total_silence = sum(silences)
    
    return {
        'silence_pct': round((total_silence / total_session_time * 100), 1) if total_session_time > 0 else 0,
        'long_pauses_count': len(long_pauses),
        'avg_pause_duration_sec': round(sum(silences) / len(silences), 2) if silences else 0
    }
```

### 4. Pitch Variation (Prosody)

- [ ] **Mean Pitch (F0)**: Average fundamental frequency in Hz (USER only)
- [ ] **Pitch Standard Deviation**: Variation in pitch (indicates expressiveness)
- [ ] **Pitch Range**: Max pitch - min pitch (Hz)

**Technical Requirements:**
- Requires `librosa` for pitch extraction
- Use `librosa.pyin()` or `librosa.piptrack()` for F0 detection
- Only analyze USER audio segments (ignore BOT)

```python
import librosa

def calculate_pitch_metrics(audio_data, sample_rate, user_segments):
    """
    Extract pitch (fundamental frequency) from USER audio.
    
    Args:
        audio_data: Audio array (mono or left channel for USER)
        sample_rate: Sample rate (typically 16000 or 48000)
        user_segments: List of {start, end} times for USER speech
    
    Returns:
        dict: Pitch metrics (mean, stddev, range)
    """
    try:
        # Extract USER audio only
        user_audio = []
        for seg in user_segments:
            start_sample = int(seg['start_time'] * sample_rate)
            end_sample = int(seg['end_time'] * sample_rate)
            user_audio.extend(audio_data[start_sample:end_sample])
        
        if len(user_audio) < 1024:
            return {'pitch_mean_hz': None, 'pitch_stddev_hz': None, 'pitch_range_hz': None}
        
        user_audio = np.array(user_audio)
        
        # Extract pitch using librosa
        f0, voiced_flag, voiced_probs = librosa.pyin(
            user_audio,
            fmin=librosa.note_to_hz('C2'),  # 65 Hz (low male voice)
            fmax=librosa.note_to_hz('C7'),  # 2093 Hz (high female voice)
            sr=sample_rate
        )
        
        # Filter out unvoiced frames (NaN values)
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) == 0:
            return {'pitch_mean_hz': None, 'pitch_stddev_hz': None, 'pitch_range_hz': None}
        
        return {
            'pitch_mean_hz': round(np.mean(f0_voiced), 1),
            'pitch_stddev_hz': round(np.std(f0_voiced), 1),
            'pitch_range_hz': round(np.max(f0_voiced) - np.min(f0_voiced), 1)
        }
        
    except Exception as e:
        error_log(f'Pitch extraction failed: {e}')
        return {'pitch_mean_hz': None, 'pitch_stddev_hz': None, 'pitch_range_hz': None}
```

### 5. Enhanced Callback Payload

- [ ] **Extend `audio_metrics` Object**: Add all new metrics
- [ ] **Backward Compatibility**: Core metrics still present (Phase 1)
- [ ] **Version Indicator**: `metrics_version: "2.0"`

**Example Payload:**
```json
{
  "session_id": "convai-abc-123",
  "transcript": "[USER]: ...",
  "audio_metrics": {
    // Phase 1 metrics (preserved)
    "speaking_pace_user_wpm": 165,
    "speaking_pace_bot_wpm": 145,
    "talk_ratio_user_pct": 48.3,
    "talk_ratio_bot_pct": 51.7,
    "total_user_time_sec": 145.2,
    "total_bot_time_sec": 155.8,
    "filler_word_count": 8,
    "filler_words_per_minute": 2.1,
    "avg_user_energy_rms": 0.72,
    "avg_response_latency_ms": 450,
    
    // Phase 2 metrics (NEW)
    "user_turn_count": 12,
    "avg_turn_length_sec": 12.1,
    "max_turn_length_sec": 34.5,
    "long_monologues_count": 2,
    "question_count_total": 8,
    "question_count_open": 5,
    "question_count_closed": 3,
    "question_ratio_open_pct": 62.5,
    "silence_pct": 8.5,
    "long_pauses_count": 3,
    "avg_pause_duration_sec": 0.45,
    "pitch_mean_hz": 180,
    "pitch_stddev_hz": 35,
    "pitch_range_hz": 120,
    
    "metrics_version": "2.0",
    "captured_at": "2025-12-02 14:30:00"
  }
}
```

---

## Error Handling

- [ ] **Graceful Degradation**: If advanced metrics fail, still return core metrics
- [ ] **Partial Success**: Include `metrics_warnings` array if some calculations fail
- [ ] **Performance**: Ensure total processing time increase <20% vs Phase 1

```python
def extract_audio_metrics_v2(word_segments, transcript, audio_data=None):
    """
    Extract all metrics (Phase 1 + Phase 2).
    """
    metrics = {}
    warnings = []
    
    try:
        # Phase 1 metrics (required)
        core_metrics = extract_core_metrics(word_segments, transcript)
        metrics.update(core_metrics)
    except Exception as e:
        error_log(f'Core metrics failed: {e}')
        return None  # Cannot proceed without core metrics
    
    try:
        # Phase 2: Turn-taking
        turn_metrics = calculate_turn_metrics(word_segments)
        metrics.update(turn_metrics)
    except Exception as e:
        warnings.append('turn_metrics_failed')
        error_log(f'Turn metrics failed: {e}')
    
    try:
        # Phase 2: Questions
        question_metrics = analyze_questions(transcript)
        metrics.update(question_metrics)
    except Exception as e:
        warnings.append('question_analysis_failed')
        error_log(f'Question analysis failed: {e}')
    
    try:
        # Phase 2: Silence
        silence_metrics = calculate_silence_metrics(word_segments, metrics['total_session_time_sec'])
        metrics.update(silence_metrics)
    except Exception as e:
        warnings.append('silence_metrics_failed')
        error_log(f'Silence metrics failed: {e}')
    
    try:
        # Phase 2: Pitch (requires audio data)
        if audio_data is not None:
            pitch_metrics = calculate_pitch_metrics(audio_data, sample_rate, user_segments)
            metrics.update(pitch_metrics)
        else:
            warnings.append('pitch_analysis_skipped_no_audio')
    except Exception as e:
        warnings.append('pitch_metrics_failed')
        error_log(f'Pitch metrics failed: {e}')
    
    metrics['metrics_version'] = '2.0'
    if warnings:
        metrics['metrics_warnings'] = warnings
    
    return metrics
```

---

## Testing Checklist

- [ ] Test turn-taking detection with varied conversation lengths
- [ ] Validate question classification accuracy (>85% correct)
- [ ] Test silence metrics with sessions containing long pauses
- [ ] Verify pitch extraction works with different voice types (male/female)
- [ ] Test error handling (missing audio, corrupted data)
- [ ] Measure performance impact (<20% slowdown vs Phase 1)
- [ ] Test backward compatibility (Phase 1 metrics still work)

---

## Dependencies

### Requires:
- **Issue #6** (Phase 1 core metrics) - Must be completed first
- `librosa` library installed in sidecar environment
- Audio data available (not just transcript)

### Enables:
- **Issue #4** (WordPress scoring algorithm) - Needs these metrics to calculate scores
- **Issue #5** (Training history scoreboard UI) - Displays computed scores

---

## Performance Considerations

**Pitch extraction is CPU-intensive:**
- Use multiprocessing if needed
- Consider async processing (return metrics via separate callback)
- Monitor sidecar processing time (should stay <30s per session)

**Fallback Strategy:**
- If pitch extraction takes >10s, skip it and return other metrics
- Log warning but don't block transcript/feedback flow

---

## Success Metrics

- ✅ All 7 new metrics extracted successfully (when data available)
- ✅ Question classification accuracy >85%
- ✅ Processing time <30s per session
- ✅ Backward compatible with Phase 1 systems
- ✅ <5% failure rate on extended metrics

---

**Created:** December 2, 2025  
**Priority:** MEDIUM (Phase 2)  
**Start After:** Issue #6 deployed and validated
