'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Mic, MicOff, Video, User, Volume2, Globe, Send, RotateCcw,
  CheckCircle, AlertCircle, BookOpen, Search, Sparkles,
  Menu, X, Info, Eye, Target, Heart,
} from 'lucide-react';
import VRMAvatar, { type VRMAAnimationInstruction } from './components/VRMAvatar';
import type { AnimationTimeline } from './components/animation/TimelinePlayer';

interface SupportedLanguage {
  code: string;
  name: string;
}

function ProgressiveTextReveal({ text, speed = 35, className = '', style = {} }: { text: string; speed?: number; className?: string; style?: React.CSSProperties }) {
  const [visibleCount, setVisibleCount] = useState(0);

  useEffect(() => {
    setVisibleCount(0);
    if (!text) return;

    const words = text.split(' ');
    let current = 0;
    const interval = setInterval(() => {
      current++;
      setVisibleCount(current);
      if (current >= words.length) {
        clearInterval(interval);
      }
    }, speed);

    return () => clearInterval(interval);
  }, [text, speed]);

  if (!text) return null;

  const words = text.split(' ');
  return (
    <span className={className} style={{ display: 'inline-block', ...style }}>
      {words.map((word, idx) => (
        <span
          key={idx}
          style={{
            display: 'inline-block',
            marginRight: '6px',
            opacity: idx < visibleCount ? 1 : 0,
            transform: idx < visibleCount ? 'translateY(0) scale(1)' : 'translateY(5px) scale(0.95)',
            transition: 'opacity 0.25s cubic-bezier(0.16, 1, 0.3, 1), transform 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
          }}
        >
          {word}
        </span>
      ))}
    </span>
  );
}

function IntroVideoOverlay({ onComplete, isFadingOut }: { onComplete: () => void; isFadingOut: boolean }) {
  const [clipIndex, setClipIndex] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const progressBarRef = useRef<HTMLDivElement>(null);

  const playlist = [
    { word: 'HELLO', url: 'http://localhost:8000/dataset/wlasl/videos/27172.mp4' },
    { word: 'WELCOME', url: 'http://localhost:8000/dataset/wlasl/videos/62837.mp4' }
  ];

  const currentClip = playlist[clipIndex];

  const handleTimeUpdate = () => {
    if (videoRef.current && videoRef.current.duration && progressBarRef.current) {
      const currentFrac = videoRef.current.currentTime / videoRef.current.duration;
      const totalProgress = ((clipIndex + currentFrac) / playlist.length) * 100;
      progressBarRef.current.style.width = `${totalProgress}%`;
    }
  };

  const handleEnded = () => {
    if (clipIndex < playlist.length - 1) {
      setClipIndex(prev => prev + 1);
    } else {
      onComplete();
    }
  };

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.play().catch(e => console.log('Intro video play error:', e));
    }
  }, [clipIndex]);

  return (
    <div className={`intro-overlay ${isFadingOut ? 'fading-out' : ''}`}>
      <div style={{
        position: 'absolute', top: '32px', left: '50%', transform: 'translateX(-50%)',
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', zIndex: 10,
        background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(20px)',
        padding: '16px 32px', borderRadius: '20px', border: '1px solid rgba(255,255,255,0.15)',
        boxShadow: '0 10px 30px rgba(0,0,0,0.5)'
      }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/samvada_logo.png" alt="Samvada Logo" style={{ height: '48px', width: '48px' }} />
        <h1 style={{ fontFamily: 'var(--font-heading)', fontSize: '20px', fontWeight: 700, color: '#fff', letterSpacing: '3px' }}>SAMVADA</h1>
        <p style={{ fontFamily: 'var(--font-inter)', fontSize: '11px', color: 'var(--color-accent-blue)', fontWeight: 500 }}>AI Sign Language & Multilingual Translation Engine</p>
      </div>

      <div style={{
        width: '85%', maxWidth: '920px', aspectRatio: '16 / 9', maxHeight: '520px',
        borderRadius: '24px', overflow: 'hidden', position: 'relative',
        boxShadow: '0 0 60px rgba(212, 168, 67, 0.4), 0 0 100px rgba(232, 145, 58, 0.25)',
        border: '1px solid rgba(255,255,255,0.2)', background: '#05040a',
        transform: 'translateZ(0)', willChange: 'opacity, transform'
      }}>
        <video
          ref={videoRef}
          src={currentClip.url}
          autoPlay
          muted
          playsInline
          preload="auto"
          onTimeUpdate={handleTimeUpdate}
          onEnded={handleEnded}
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
        />
        
        {/* ASL Sign Concept Badge */}
        <div style={{
          position: 'absolute', top: '20px', right: '20px',
          background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(10px)',
          color: '#38bdf8', padding: '6px 16px', borderRadius: '10px',
          fontSize: '13px', fontWeight: 700, border: '1px solid rgba(56,189,248,0.3)',
          letterSpacing: '0.5px'
        }}>
          ASL SIGN: "{currentClip.word}" ({clipIndex + 1}/{playlist.length})
        </div>

        {/* Progress Bar Container */}
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0, height: '4px', background: 'rgba(255,255,255,0.1)'
        }}>
          <div 
            ref={progressBarRef}
            style={{ width: '0%', height: '100%', background: 'linear-gradient(90deg, #d4a843, #e8913a)' }} 
          />
        </div>
      </div>

      <div style={{ marginTop: '28px', display: 'flex', alignItems: 'center', gap: '16px', zIndex: 10 }}>
        <button
          onClick={onComplete}
          className="glass-btn glass-btn-primary intro-badge-glow"
          style={{ padding: '12px 32px', fontSize: '14px', borderRadius: '14px', fontWeight: 700 }}
        >
          Skip & Explore SAMVADA Dashboard →
        </button>
      </div>
    </div>
  );
}

interface DictionaryMeaning {
  partOfSpeech: string;
  definitions: string[];
}

interface LearnedWord {
  phonetic?: string;
  meanings: DictionaryMeaning[];
}

interface RecognitionSocketMessage {
  code?: string;
  error?: string;
  type?: 'fingerspell' | 'word' | 'landmarks' | 'uncertain' | 'error' | 'sentence';
  prediction?: string;
  confidence?: number;
  confidence_threshold?: number;
  alternatives?: Array<{ label: string; confidence: number }>;
  glosses?: string[];
  english_text?: string;
  translated_text?: string;
  translation?: string;
  message?: string;
  left_hand?: number[][];
  right_hand?: number[][];
  left_palm_normal?: number[];
  right_palm_normal?: number[];
  detector_latency_ms?: number;
  has_lh?: boolean;
  has_rh?: boolean;
}

export default function Home() {
  // Navigation & Page state (Intro plays automatically on website startup)
  const [showIntro, setShowIntro] = useState<boolean>(true);
  const [introFadingOut, setIntroFadingOut] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'textToSign' | 'signToText'>('textToSign');
  const [languages, setLanguages] = useState<SupportedLanguage[]>([]);
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [sourceLanguage, setSourceLanguage] = useState('auto');

  const handleDismissIntro = useCallback(() => {
    setIntroFadingOut(true);
    setTimeout(() => {
      setShowIntro(false);
      setIntroFadingOut(false);
    }, 800);
  }, []);

  // Toasts, Diagnostics & Webcam Status States
  interface Toast {
    id: string;
    message: string;
    type: 'success' | 'error' | 'info' | 'warning';
  }
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  
  interface DiagnosticMetric {
    count: number;
    average_latency_ms: number;
    success_rate: number;
  }
  const [diagnosticsData, setDiagnosticsData] = useState<Record<string, DiagnosticMetric>>({});
  const [modelStates, setModelStates] = useState<Record<string, { available: boolean; error: string | null }>>({});
  const [webcamStatus, setWebcamStatus] = useState<'off' | 'starting' | 'connected'>('off');
  const [showWireframe, setShowWireframe] = useState(true);

  const showToast = useCallback((message: string, type: 'success' | 'error' | 'info' | 'warning' = 'info') => {
    const id = window.crypto.randomUUID();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 4000);
  }, []);

  // Text -> Sign State
  const [inputText, setInputText] = useState('');
  const [translating, setTranslating] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [displayMode, setDisplayMode] = useState<'video' | 'avatar'>('video');
  
  // We track the previous text to avoid redundant translations
  const [prevInputText, setPrevInputText] = useState('');
  
  // Ref to prevent auto-translate when mic sets the text
  const skipAutoTranslateRef = useRef(false);
  
  // Translation Results
  const [glosses, setGlosses] = useState<string[]>([]);
  const [translatedCaption, setTranslatedCaption] = useState('');
  type VideoEntry = string | { type: 'text'; word: string; duration: number } | { type: 'image'; word: string; url: string; duration?: number };
  const [videoSequence, setVideoSequence] = useState<VideoEntry[]>([]);
  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);
  const [animationTimeline, setAnimationTimeline] = useState<AnimationTimeline | undefined>(undefined);
  const [avatarAnimations, setAvatarAnimations] = useState<VRMAAnimationInstruction[]>([]);
  const [currentAnimationIndex, setCurrentAnimationIndex] = useState(0);
  
  // Playback Controls
  const [playbackSpeed, setPlaybackSpeed] = useState(1.25);

  // Sign -> Text State
  const [recognitionMode, setRecognitionMode] = useState<'word' | 'fingerspell' | 'sentence'>('word');
  const [recognizedGlosses, setRecognizedGlosses] = useState<string[]>([]);
  const [recognizedText, setRecognizedText] = useState('');
  const [translatedOutput, setTranslatedOutput] = useState('');
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [webcamError, setWebcamError] = useState('');

  // Dynamic Word Learning State
  const [newWordToLearn, setNewWordToLearn] = useState('');
  const [learningResult, setLearningResult] = useState<LearnedWord | null>(null);
  const [learningError, setLearningError] = useState('');
  const [learningLoading, setLearningLoading] = useState(false);

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const webcamRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const socketRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const liveLandmarksRef = useRef<{
    left_hand: number[][];
    right_hand: number[][];
    left_palm_normal: number[];
    right_palm_normal: number[];
    detector_latency_ms: number;
  }>({
    left_hand: [],
    right_hand: [],
    left_palm_normal: [0, 0, 1],
    right_palm_normal: [0, 0, 1],
    detector_latency_ms: 0
  });

  // Refs to avoid re-triggering the WebSocket/MediaPipe pipeline on mode/language changes
  const recognitionModeRef = useRef(recognitionMode);
  const selectedLanguageRef = useRef(selectedLanguage);
  const idleSpaceTimerRef = useRef<any>(null);
  useEffect(() => { recognitionModeRef.current = recognitionMode; }, [recognitionMode]);
  useEffect(() => { selectedLanguageRef.current = selectedLanguage; }, [selectedLanguage]);

  // Reset output text and glosses when switching between Word and Fingerspell modes
  useEffect(() => {
    setRecognizedText('');
    setRecognizedGlosses([]);
    setTranslatedOutput('');
    if (idleSpaceTimerRef.current) {
      clearTimeout(idleSpaceTimerRef.current);
      idleSpaceTimerRef.current = null;
    }
  }, [recognitionMode]);

  // Global Keyboard Navigation Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeElement = document.activeElement as HTMLElement | null;
      const tag = activeElement?.tagName?.toUpperCase();
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || activeElement?.isContentEditable) {
        return;
      }

      if (e.code === 'Space' || e.code === 'KeyC') {
        e.preventDefault();
        setIsWebcamActive(prev => !prev);
      } else if (e.code === 'KeyM') {
        e.preventDefault();
        // Trigger microphone recording toggle
        const micBtn = document.getElementById('btn-mic-toggle');
        if (micBtn) micBtn.click();
      } else if (e.code === 'Digit1') {
        e.preventDefault();
        setActiveTab('textToSign');
      } else if (e.code === 'Digit2') {
        e.preventDefault();
        setActiveTab('signToText');
      } else if (e.code === 'KeyD') {
        e.preventDefault();
        setShowDiagnostics(prev => !prev);
      } else if (e.code === 'Escape') {
        e.preventDefault();
        setToasts([]);
        setShowDiagnostics(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const getSessionId = () => {
    if (!sessionIdRef.current) {
      sessionIdRef.current = window.crypto.randomUUID();
    }
    return sessionIdRef.current;
  };

  // Fetch diagnostics utility
  const fetchDiagnostics = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8000/api/diagnostics');
      if (res.ok) {
        const data = await res.json();
        setDiagnosticsData(data.operations || {});
        setModelStates(data.models || {});
      }
    } catch (err) {
      console.error("Failed to fetch diagnostics", err);
    }
  }, []);

  // 1. Check Backend connection, load languages and setup reconnection loop
  useEffect(() => {
    let active = true;
    const checkBackend = async () => {
      try {
        const res = await fetch('http://localhost:8000/');
        if (res.ok) {
          if (active) {
            setBackendOnline(true);
            // Fetch languages list
            const langRes = await fetch('http://localhost:8000/api/languages');
            const langData = await langRes.json() as { languages?: SupportedLanguage[] };
            setLanguages(langData.languages || []);
            fetchDiagnostics();
          }
        } else {
          if (active) setBackendOnline(false);
        }
      } catch {
        if (active) setBackendOnline(false);
      }
    };

    checkBackend();

    // Reconnection poll when offline
    const interval = setInterval(async () => {
      if (!backendOnline) {
        try {
          const res = await fetch('http://localhost:8000/');
          if (res.ok) {
            setBackendOnline(true);
            showToast("Backend connection re-established!", "success");
            const langRes = await fetch('http://localhost:8000/api/languages');
            const langData = await langRes.json() as { languages?: SupportedLanguage[] };
            setLanguages(langData.languages || []);
            fetchDiagnostics();
          }
        } catch {
          // keep polling
        }
      } else {
        // Poll diagnostics only while online
        fetchDiagnostics();
      }
    }, 5000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [backendOnline, fetchDiagnostics, showToast]);

  // Apple VisionOS Dynamic Pointer Light Tracking Effect
  useEffect(() => {
    let animationFrameId: number;
    const handleMouseMove = (e: MouseEvent) => {
      animationFrameId = requestAnimationFrame(() => {
        const elements = document.querySelectorAll('.glass-btn, .glass-input');
        elements.forEach((el) => {
          const rect = el.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;
          (el as HTMLElement).style.setProperty('--mouse-x', `${x}px`);
          (el as HTMLElement).style.setProperty('--mouse-y', `${y}px`);
        });
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  // 2. Play Video Sequence one after another (supports text placeholders and same-URL replay)
  const prevVideoUrlRef = useRef<string>('');
  const ttsPlayedForSequenceRef = useRef<boolean>(false);

  useEffect(() => {
    if (displayMode !== 'video' || videoSequence.length === 0) return;

    // Reset TTS tracking flag when a new sequence starts
    if (currentVideoIndex === 0) {
      if (!ttsPlayedForSequenceRef.current && translatedCaption) {
        handleTextToSpeech(translatedCaption, selectedLanguage);
        ttsPlayedForSequenceRef.current = true;
      }
    } else {
      // Just in case we jump around, although normally it resets on new translation
    }

    const entry = videoSequence[currentVideoIndex];
    const advance = () => {
      if (currentVideoIndex < videoSequence.length - 1) {
        setCurrentVideoIndex(prev => prev + 1);
      }
    };

    // Handle text or image placeholder entries (e.g. ASL alphabet letter cards)
    if (typeof entry === 'object' && (entry.type === 'text' || entry.type === 'image')) {
      const timer = setTimeout(advance, (entry.duration || 1.5) * 1000);
      return () => clearTimeout(timer);
    }

    // Handle video entries
    if (videoRef.current && typeof entry === 'string') {
      const video = videoRef.current;
      const newUrl = `http://localhost:8000/${entry}`;
      
      // If same URL as previous, just replay from start
      if (newUrl === prevVideoUrlRef.current) {
        video.currentTime = 0;
        video.play().catch(e => console.log("Video replay blocked", e));
      } else {
        video.src = newUrl;
        prevVideoUrlRef.current = newUrl;
        video.playbackRate = playbackSpeed;
        video.load();
        video.play().catch(e => console.log("Video autoplay blocked", e));
      }
      
      const handleEnded = () => advance();
      video.addEventListener('ended', handleEnded);
      return () => video.removeEventListener('ended', handleEnded);
    }
  }, [videoSequence, currentVideoIndex, displayMode, playbackSpeed]);

  // 3. Advance only after the authored VRMA reports that it has finished.
  // Fixed-duration timers desynchronise from source animation data.
  const handleAvatarAnimationFinished = useCallback(() => {
    setCurrentAnimationIndex((currentIndex) => {
      if (avatarAnimations.length === 0) return 0;
      if (currentIndex < avatarAnimations.length - 1) {
        return currentIndex + 1;
      }
      return currentIndex;
    });
  }, [avatarAnimations.length]);

  // Auto-advance text_hold avatar animations after their duration
  useEffect(() => {
    if (avatarAnimations.length === 0) return;
    const current = avatarAnimations[currentAnimationIndex];
    if (current && current.type === 'text_hold') {
      const duration = (current as {duration?: number}).duration || 2.0;
      const timer = setTimeout(() => {
        handleAvatarAnimationFinished();
      }, duration * 1000);
      return () => clearTimeout(timer);
    }
  }, [currentAnimationIndex, avatarAnimations, handleAvatarAnimationFinished]);

  // 3. Text Input & Auto-Translate (Debounced) — also re-translates on language switch
  const prevLangRef = useRef(selectedLanguage);
  useEffect(() => {
    if (!isRecording) {
      const languageChanged = prevLangRef.current !== selectedLanguage;
      prevLangRef.current = selectedLanguage;

      // If language changed mid-translation, immediately cancel audio and reset sequence
      if (languageChanged) {
        try {
          if (currentAudioRef.current) {
            currentAudioRef.current.pause();
            currentAudioRef.current.src = "";
            currentAudioRef.current = null;
          }
          if (typeof window !== 'undefined' && window.speechSynthesis) {
            window.speechSynthesis.cancel();
          }
        } catch { /* ignore */ }
        setIsSpeaking(false);
        setGlosses([]);
        setVideoSequence([]);
        setCurrentVideoIndex(0);
        setAvatarAnimations([]);
        setCurrentAnimationIndex(0);
        ttsPlayedForSequenceRef.current = false;
        setTranslatedCaption('');
      }

      const timer = setTimeout(() => {
        if (skipAutoTranslateRef.current) {
          skipAutoTranslateRef.current = false;
          return;
        }

        // Re-translate if text exists AND (text changed OR language changed)
        if (inputText.trim() && (inputText !== translatedCaption || languageChanged)) {
          translateTextToSign(inputText);
        }
      }, languageChanged ? 50 : 300); // Near-instant on language switch, debounced on typing
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputText, selectedLanguage, sourceLanguage, isRecording]);

  // 4. Speech to Text (MediaRecorder + Backend Whisper integration)
  const handleMicrophoneToggle = async () => {
    if (isRecording) {
      // Stop recording
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
        mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      }
      setIsRecording(false);
    } else {
      // Start recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            noiseSuppression: true,
            echoCancellation: true,
            autoGainControl: true
          } 
        });
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];
        
        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };
        
        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          audioChunksRef.current = [];
          
          if (audioBlob.size === 0) return;
          
          // Use toast instead of input text to prevent auto-translate triggering
          // showToast("Processing audio...", "info");
          setTranslating(true);
          
          try {
            const formData = new FormData();
            formData.append("audio", audioBlob, "recording.webm");
            formData.append("target_lang", selectedLanguage);
            formData.append("source_lang", sourceLanguage);
            
            const response = await fetch('http://localhost:8000/api/translate/speech-to-text', {
              method: 'POST',
              body: formData
            });
            
            if (response.ok) {
              const data = await response.json();
              if (data.success) {
                // Prevent the text box debounce from triggering a second identical translation
                skipAutoTranslateRef.current = true;
                
                // Set the translated text without triggering TTS (the video playback will trigger TTS automatically when it loads)
                setInputText(data.english_text); // The translated English text
                setTranslatedCaption(data.translated_text || data.english_text);
                showToast("Speech transcribed successfully!", "success");

                if (data.sign_language) {
                  setGlosses(data.sign_language.glosses || []);
                  setVideoSequence(data.sign_language.videos || []);
                  ttsPlayedForSequenceRef.current = false;
                  setCurrentVideoIndex(0);
                  setAvatarAnimations(data.sign_language.avatar_animations || []);
                  setCurrentAnimationIndex(0);
                }
              } else {
                setInputText("");
                showToast("Transcription failed: " + data.error, "error");
              }
            } else {
              const errData = await response.json();
              console.error("Backend speech recognition error:", errData);
              showToast("Failed to transcribe audio on the server.", "error");
              setInputText("");
            }
          } catch (error) {
            console.error("Error sending audio to backend:", error);
            showToast("Network error: Failed to connect to transcription server.", "error");
            setInputText("");
          } finally {
            setTranslating(false);
          }
        };
        
        mediaRecorder.start();
        setIsRecording(true);
        // showToast("Listening...", "info");
        
      } catch (err) {
        console.error("Microphone access failed", err);
        showToast("Microphone permission is strictly required. Please check your browser site settings.", "warning");
      }
    }
  };

  // 5. Submit Text for translation to Sign
  const translateTextToSign = async (textToTranslate = inputText) => {
    const text = textToTranslate.trim();
    if (!text) return;
    
    setTranslating(true);
    try {
      const response = await fetch('http://localhost:8000/api/translate/text-to-sign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          target_lang: selectedLanguage,
          source_lang: sourceLanguage,
          session_id: getSessionId(),
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setGlosses(data.glosses || []);
        
        const outputCaption = data.caption_text || '';
        setTranslatedCaption(outputCaption);
        
        setVideoSequence(data.videos || []);
        ttsPlayedForSequenceRef.current = false;
        setCurrentVideoIndex(0);
        setAnimationTimeline(data.animation_timeline || undefined);
        setAvatarAnimations(data.avatar_animations || []);
        setCurrentAnimationIndex(0);
      }
    } catch (err) {
      console.error("Translation failed", err);
    } finally {
      setTranslating(false);
    }
  };

  // 6. Webcam + WebSocket setup for Sign -> Text
  useEffect(() => {
    let animationFrameId: number;
    const webcamElement = webcamRef.current;
    
    if (isWebcamActive && webcamRef.current) {
      setWebcamStatus('starting');
      // Connect to WebSocket
      const socket = new WebSocket('ws://localhost:8000/api/ws/recognize');
      socketRef.current = socket;
      
      socket.onopen = () => {
        console.info('[Frontend] Webcam WebSocket connection opened.');
        showToast("Connected to Live Sign Recognition engine.", "success");
        setWebcamStatus('connected');
      };
      
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data) as RecognitionSocketMessage;
        if (data.error) {
          console.error(data.error);
          showToast(data.error, "error");
          return;
        }
        
        if (data.type === 'landmarks') {
          liveLandmarksRef.current = {
            left_hand: data.left_hand || [],
            right_hand: data.right_hand || [],
            left_palm_normal: data.left_palm_normal || [0, 0, 1],
            right_palm_normal: data.right_palm_normal || [0, 0, 1],
            detector_latency_ms: data.detector_latency_ms || 0
          };
        } else if (data.type === 'fingerspell') {
          // Append fingerspell letter
          const letter = data.prediction;
          if (!letter) return;

          if (idleSpaceTimerRef.current) {
            clearTimeout(idleSpaceTimerRef.current);
            idleSpaceTimerRef.current = null;
          }

          if (letter === 'space' || letter === 'SPACE') {
            setRecognizedText(prev => (prev.endsWith(' ') ? prev : prev + ' '));
          } else if (letter !== 'nothing' && letter !== 'NOTHING' && letter !== 'del' && letter !== 'DEL') {
            setRecognizedText(prev => prev + letter);
          }

          // Automatically insert a space after 3 seconds of pause
          idleSpaceTimerRef.current = setTimeout(() => {
            setRecognizedText(prev => (prev && !prev.endsWith(' ') ? prev + ' ' : prev));
          }, 3000);
        } else if (data.type === 'word') {
          setRecognizedGlosses(data.glosses || []);
          setRecognizedText(data.english_text || data.prediction || '');
          const targetText = data.translated_text || data.translation || data.english_text || '';
          setTranslatedOutput(targetText);

          // Auto-trigger TTS exactly 1 second (1000ms) after translated text occurs
          if (targetText && targetText !== 'NOTHING') {
            setTimeout(() => {
              handleTextToSpeech(targetText, selectedLanguage);
            }, 1000);
          }
        } else if (data.type === 'uncertain') {
          // Show top candidate as live scanning indicator
          if (data.alternatives && data.alternatives.length > 0) {
            const top = data.alternatives[0];
            setRecognizedText(`Scanning: ${top.label} (${Math.round(top.confidence * 100)}%)`);
          }
        } else if (data.type === 'sentence') {
          setRecognizedText(data.english_text || data.prediction || '');
          const targetText = data.translated_text || data.english_text || '';
          setTranslatedOutput(targetText);
          if (targetText && targetText !== 'NOTHING') {
            setTimeout(() => {
              handleTextToSpeech(targetText, selectedLanguage);
            }, 1000);
          }
        }
      };
      
      socket.onclose = () => {
        console.info('[Frontend] Webcam WebSocket connection closed.');
        if (idleSpaceTimerRef.current) {
          clearTimeout(idleSpaceTimerRef.current);
          idleSpaceTimerRef.current = null;
        }
        setWebcamStatus('off');
      };

      // ═══════════════════════════════════════════════════════════════════
      // 3D Adaptive 1-Euro Filter for zero-jitter, zero-lag landmark rendering
      // Dynamically scales cutoff frequency based on velocity:
      //   Low velocity (static poses) → heavy smoothing → zero jitter
      //   High velocity (fast motion) → light smoothing → zero phase lag
      // ═══════════════════════════════════════════════════════════════════
      class OneEuroFilter3D {
        private minCutoff: number;
        private beta: number;
        private dCutoff: number;
        private xPrev: number[][] | null = null;
        private dxPrev: number[][] | null = null;
        private tPrev: number | null = null;

        constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
          this.minCutoff = minCutoff;
          this.beta = beta;
          this.dCutoff = dCutoff;
        }

        private alpha(dt: number, cutoff: number): number {
          const r = 2 * Math.PI * cutoff * dt;
          return r / (r + 1.0);
        }

        filter(x: number[][], timestamp: number): number[][] {
          if (!this.xPrev || !this.dxPrev || this.tPrev === null || this.xPrev.length !== x.length) {
            this.xPrev = x.map(p => [...p]);
            this.dxPrev = x.map(() => [0, 0, 0]);
            this.tPrev = timestamp;
            return x.map(p => [...p]);
          }

          const dt = Math.max((timestamp - this.tPrev) / 1000, 0.0001);
          this.tPrev = timestamp;
          const aD = this.alpha(dt, this.dCutoff);

          const result: number[][] = [];
          for (let i = 0; i < x.length; i++) {
            const dx = x[i].map((v, j) => (v - this.xPrev![i][j]) / dt);
            const dxHat = dx.map((v, j) => aD * v + (1 - aD) * this.dxPrev![i][j]);
            const speed = Math.sqrt(dxHat[0] ** 2 + dxHat[1] ** 2 + (dxHat[2] || 0) ** 2);
            const cutoff = this.minCutoff + this.beta * speed;
            const a = this.alpha(dt, cutoff);
            const xHat = x[i].map((v, j) => a * v + (1 - a) * this.xPrev![i][j]);
            this.dxPrev![i] = dxHat;
            this.xPrev![i] = xHat;
            result.push(xHat);
          }
          return result;
        }

        reset() { this.xPrev = null; this.dxPrev = null; this.tPrev = null; }
      }

      // Instantiate independent filters for each tracked body part
      const leftHandFilter = new OneEuroFilter3D(1.5, 0.04, 1.0);
      const rightHandFilter = new OneEuroFilter3D(1.5, 0.04, 1.0);
      const poseFilter = new OneEuroFilter3D(1.2, 0.02, 1.0);
      const smoothedLeftRef: { current: number[][] } = { current: [] };
      const smoothedRightRef: { current: number[][] } = { current: [] };
      const smoothedPoseRef: { current: number[][] } = { current: [] };

      // Initialize Client-Side MediaPipe Tasks Engine (WebGL 60 FPS)
      let handLandmarker: any = null;
      let poseLandmarker: any = null;
      let mediapipeReady = false;

      if (typeof window !== 'undefined') {
        console.info('[Frontend] Downloading MediaPipe Vision models (~15MB)...');
        import('@mediapipe/tasks-vision')
          .then(async (visionModule) => {
            try {
              const vision = await visionModule.FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
              );
              handLandmarker = await visionModule.HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                  modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                  delegate: 'CPU'
                },
                runningMode: 'VIDEO',
                numHands: 2
              });
              poseLandmarker = await visionModule.PoseLandmarker.createFromOptions(vision, {
                baseOptions: {
                  modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
                  delegate: 'CPU'
                },
                runningMode: 'VIDEO',
                numPoses: 1
              });
              mediapipeReady = true;
              console.info('[Frontend] Client-Side WebGL MediaPipe Tasks Engine initialized successfully!');
              showToast('Vision AI models loaded — landmark tracking active!', 'success');
            } catch (err) {
              console.warn('[Frontend] Client MediaPipe GPU init failed, using server-side detection:', err);
              showToast('Using server-side landmark detection (GPU init failed)', 'warning');
            }
          })
          .catch(e => {
            console.warn('[Frontend] Dynamic import MediaPipe tasks vision failed:', e);
            showToast('Using server-side landmark detection', 'info');
          });
      }

      // Instant-open webcam: start with zero constraints (native resolution, <200ms),
      // then silently upgrade to 1280x720 in the background after the feed is already live.
      navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
        .then((stream) => {
          if (webcamRef.current) {
            webcamRef.current.srcObject = stream;
            webcamRef.current.play().catch(e => console.log(e));
          }

          // Background resolution upgrade — user already sees the feed
          const videoTrack = stream.getVideoTracks()[0];
          if (videoTrack) {
            videoTrack.applyConstraints({ width: { ideal: 1280 }, height: { ideal: 720 } })
              .then(() => console.info('[Frontend] Webcam upgraded to 1280x720'))
              .catch(() => console.info('[Frontend] Camera stays at native resolution'));
          }

          let lastWsSendTime = 0;
          const renderAndSend = (timestamp: number) => {
            if (webcamRef.current && webcamRef.current.readyState >= 2) {
              const video = webcamRef.current;
              const now = performance.now();

              let clientLeft: number[][] = [];
              let clientRight: number[][] = [];
              let clientPose: number[][] = [];

              // 1. Client-Side WebGL 60 FPS Landmark Detection (only when models loaded)
              if (mediapipeReady && handLandmarker && poseLandmarker) {
                try {
                  const handRes = handLandmarker.detectForVideo(video, now);
                  const poseRes = poseLandmarker.detectForVideo(video, now);

                  if (handRes.landmarks && handRes.handedness) {
                    handRes.landmarks.forEach((lms: any[], idx: number) => {
                      const category = handRes.handedness[idx]?.[0]?.categoryName?.toLowerCase();
                      const pts = lms.map((p: any) => [p.x, p.y, p.z || 0]);
                      if (category === 'left') {
                        clientLeft = pts;
                      } else {
                        clientRight = pts;
                      }
                    });
                  }

                  if (poseRes.landmarks && poseRes.landmarks.length > 0) {
                    clientPose = poseRes.landmarks[0].map((p: any) => [p.x, p.y, p.z || 0]);
                  }
                } catch (detErr) {
                  // Fallback to server landmark stream if detection is skipped
                }
              }

              // Only render client-side detected landmarks — no stale fallback
              const rawLeft = clientLeft.length === 21 ? clientLeft : null;
              const rawRight = clientRight.length === 21 ? clientRight : null;
              const rawPose = clientPose.length > 0 ? clientPose : null;

              // 2. Render Smooth 60 FPS Pose + Hand Skeletons
              if (overlayCanvasRef.current) {
                const oCanvas = overlayCanvasRef.current;
                if (video.videoWidth > 0 && video.videoHeight > 0) {
                  if (oCanvas.width !== video.videoWidth || oCanvas.height !== video.videoHeight) {
                    oCanvas.width = video.videoWidth;
                    oCanvas.height = video.videoHeight;
                  }
                }
                const oCtx = oCanvas.getContext('2d');
                if (oCtx) {
                  const w = oCanvas.width;
                  const h = oCanvas.height;
                  oCtx.clearRect(0, 0, w, h);

                  // Corner Brackets
                  oCtx.strokeStyle = 'rgba(212, 168, 67, 0.7)';
                  oCtx.lineWidth = 2;
                  const margin = 12;
                  const len = 18;
                  oCtx.beginPath(); oCtx.moveTo(margin, margin + len); oCtx.lineTo(margin, margin); oCtx.lineTo(margin + len, margin); oCtx.stroke();
                  oCtx.beginPath(); oCtx.moveTo(w - margin - len, margin); oCtx.lineTo(w - margin, margin); oCtx.lineTo(w - margin, margin + len); oCtx.stroke();
                  oCtx.beginPath(); oCtx.moveTo(margin, h - margin - len); oCtx.lineTo(margin, h - margin); oCtx.lineTo(margin + len, h - margin); oCtx.stroke();
                  oCtx.beginPath(); oCtx.moveTo(w - margin - len, h - margin); oCtx.lineTo(w - margin, h - margin); oCtx.lineTo(w - margin, h - margin - len); oCtx.stroke();

                  // Animated Scanline
                  const scanY = (timestamp % 1800) / 1800 * (h - margin * 2) + margin;
                  oCtx.strokeStyle = 'rgba(232, 145, 58, 0.45)';
                  oCtx.lineWidth = 1.5;
                  oCtx.beginPath(); oCtx.moveTo(margin, scanY); oCtx.lineTo(w - margin, scanY); oCtx.stroke();

                  // A. Draw Pose Skeleton (Shoulders, Elbows, Wrists)
                  if (rawPose && rawPose.length >= 17) {
                    const sL = rawPose[11];
                    const sR = rawPose[12];
                    const eL = rawPose[13];
                    const eR = rawPose[14];
                    const wL = rawPose[15];
                    const wR = rawPose[16];

                    oCtx.strokeStyle = 'rgba(212, 168, 67, 0.85)';
                    oCtx.lineWidth = 3.5;

                    if (sL && eL && wL) {
                      oCtx.beginPath();
                      oCtx.moveTo(sL[0] * w, sL[1] * h);
                      oCtx.lineTo(eL[0] * w, eL[1] * h);
                      oCtx.lineTo(wL[0] * w, wL[1] * h);
                      oCtx.stroke();
                    }
                    if (sR && eR && wR) {
                      oCtx.beginPath();
                      oCtx.moveTo(sR[0] * w, sR[1] * h);
                      oCtx.lineTo(eR[0] * w, eR[1] * h);
                      oCtx.lineTo(wR[0] * w, wR[1] * h);
                      oCtx.stroke();
                    }
                    if (sL && sR) {
                      oCtx.beginPath();
                      oCtx.moveTo(sL[0] * w, sL[1] * h);
                      oCtx.lineTo(sR[0] * w, sR[1] * h);
                      oCtx.stroke();
                    }

                    // Pose Joint Nodes — color-coded by body part
                    const poseJoints: [number[] | undefined, string][] = [
                      [sL, '#38bdf8'], [sR, '#38bdf8'],   // Shoulders — sky blue
                      [eL, '#a78bfa'], [eR, '#a78bfa'],   // Elbows — lavender
                      [wL, '#fbbf24'], [wR, '#fbbf24'],   // Wrists — gold
                    ];
                    poseJoints.forEach(([p, color]) => {
                      if (p) {
                        oCtx.fillStyle = color;
                        oCtx.beginPath();
                        oCtx.arc(p[0] * w, p[1] * h, 5.5, 0, Math.PI * 2);
                        oCtx.fill();
                      }
                    });

                    // Nose tracking & face outline
                    const nose = rawPose[0];
                    if (nose) {
                      // Draw nose point as bright white dot
                      oCtx.fillStyle = '#ffffff';
                      oCtx.beginPath();
                      oCtx.arc(nose[0] * w, nose[1] * h, 6, 0, Math.PI * 2);
                      oCtx.fill();

                      // Lines from nose to each shoulder
                      oCtx.strokeStyle = 'rgba(212, 168, 67, 0.65)';
                      oCtx.lineWidth = 2.5;
                      if (sL) {
                        oCtx.beginPath();
                        oCtx.moveTo(nose[0] * w, nose[1] * h);
                        oCtx.lineTo(sL[0] * w, sL[1] * h);
                        oCtx.stroke();
                      }
                      if (sR) {
                        oCtx.beginPath();
                        oCtx.moveTo(nose[0] * w, nose[1] * h);
                        oCtx.lineTo(sR[0] * w, sR[1] * h);
                        oCtx.stroke();
                      }
                    }
                  }

                  // B. Draw Hand Skeleton with Per-Finger Color Differentiation
                  const FINGER_GROUPS = [
                    { name: 'thumb',  indices: [0,1,2,3,4],     connections: [[0,1],[1,2],[2,3],[3,4]] },
                    { name: 'index',  indices: [5,6,7,8],       connections: [[0,5],[5,6],[6,7],[7,8]] },
                    { name: 'middle', indices: [9,10,11,12],     connections: [[0,9],[9,10],[10,11],[11,12]] },
                    { name: 'ring',   indices: [13,14,15,16],    connections: [[0,13],[13,14],[14,15],[15,16]] },
                    { name: 'pinky',  indices: [17,18,19,20],    connections: [[0,17],[17,18],[18,19],[19,20]] },
                  ];
                  const PALM_CONNECTIONS = [[5,9],[9,13],[13,17]];

                  // Left Hand — Cool Palette (Blues & Cyans)
                  const LEFT_COLORS: Record<string, { dot: string; line: string }> = {
                    thumb:  { dot: '#00e5ff', line: 'rgba(0, 229, 255, 0.85)' },
                    index:  { dot: '#40c4ff', line: 'rgba(64, 196, 255, 0.85)' },
                    middle: { dot: '#448aff', line: 'rgba(68, 138, 255, 0.85)' },
                    ring:   { dot: '#536dfe', line: 'rgba(83, 109, 254, 0.85)' },
                    pinky:  { dot: '#7c4dff', line: 'rgba(124, 77, 255, 0.85)' },
                  };
                  const LEFT_WRIST = '#18ffff';
                  const LEFT_FINGERTIP = '#e0f7fa';
                  const LEFT_PALM_LINE = 'rgba(100, 180, 255, 0.6)';

                  // Right Hand — Warm Palette (Reds & Oranges)
                  const RIGHT_COLORS: Record<string, { dot: string; line: string }> = {
                    thumb:  { dot: '#ff1744', line: 'rgba(255, 23, 68, 0.85)' },
                    index:  { dot: '#ff9100', line: 'rgba(255, 145, 0, 0.85)' },
                    middle: { dot: '#ffea00', line: 'rgba(255, 234, 0, 0.85)' },
                    ring:   { dot: '#ff6d00', line: 'rgba(255, 109, 0, 0.85)' },
                    pinky:  { dot: '#ff4081', line: 'rgba(255, 64, 129, 0.85)' },
                  };
                  const RIGHT_WRIST = '#ffd740';
                  const RIGHT_FINGERTIP = '#fff8e1';
                  const RIGHT_PALM_LINE = 'rgba(255, 160, 100, 0.6)';

                  const FINGERTIP_INDICES = new Set([4, 8, 12, 16, 20]);

                  const drawColoredHand = (pts: number[][], palette: Record<string, { dot: string; line: string }>, wristColor: string, tipColor: string, palmLine: string) => {
                    if (!pts || pts.length < 21) return;

                    // Draw finger connections with per-finger colors
                    FINGER_GROUPS.forEach(fg => {
                      oCtx.strokeStyle = palette[fg.name].line;
                      oCtx.lineWidth = 2.5;
                      fg.connections.forEach(([i, j]) => {
                        const p1 = pts[i]; const p2 = pts[j];
                        if (p1 && p2) {
                          oCtx.beginPath();
                          oCtx.moveTo(p1[0] * w, p1[1] * h);
                          oCtx.lineTo(p2[0] * w, p2[1] * h);
                          oCtx.stroke();
                        }
                      });
                    });

                    // Draw palm base connections
                    oCtx.strokeStyle = palmLine;
                    oCtx.lineWidth = 2;
                    PALM_CONNECTIONS.forEach(([i, j]) => {
                      const p1 = pts[i]; const p2 = pts[j];
                      if (p1 && p2) {
                        oCtx.beginPath();
                        oCtx.moveTo(p1[0] * w, p1[1] * h);
                        oCtx.lineTo(p2[0] * w, p2[1] * h);
                        oCtx.stroke();
                      }
                    });

                    // Draw joint nodes with finger-specific colors
                    pts.forEach((p, idx) => {
                      if (idx === 0) {
                        // Wrist — distinct large highlight
                        oCtx.fillStyle = wristColor;
                        oCtx.beginPath();
                        oCtx.arc(p[0] * w, p[1] * h, 6.5, 0, Math.PI * 2);
                        oCtx.fill();
                      } else if (FINGERTIP_INDICES.has(idx)) {
                        // Fingertips — glowing highlight
                        oCtx.fillStyle = tipColor;
                        oCtx.beginPath();
                        oCtx.arc(p[0] * w, p[1] * h, 5.5, 0, Math.PI * 2);
                        oCtx.fill();
                      } else {
                        // Regular joints — finger color
                        const finger = idx <= 4 ? 'thumb' : idx <= 8 ? 'index' : idx <= 12 ? 'middle' : idx <= 16 ? 'ring' : 'pinky';
                        oCtx.fillStyle = palette[finger].dot;
                        oCtx.beginPath();
                        oCtx.arc(p[0] * w, p[1] * h, 4.0, 0, Math.PI * 2);
                        oCtx.fill();
                      }
                    });
                  };

                  // C. Apply 3D 1-Euro Adaptive Filtering & Draw Hands
                  if (rawLeft) {
                    smoothedLeftRef.current = leftHandFilter.filter(rawLeft, timestamp);
                    drawColoredHand(smoothedLeftRef.current, LEFT_COLORS, LEFT_WRIST, LEFT_FINGERTIP, LEFT_PALM_LINE);
                  } else {
                    smoothedLeftRef.current = [];
                    leftHandFilter.reset();
                  }

                  if (rawRight) {
                    smoothedRightRef.current = rightHandFilter.filter(rawRight, timestamp);
                    drawColoredHand(smoothedRightRef.current, RIGHT_COLORS, RIGHT_WRIST, RIGHT_FINGERTIP, RIGHT_PALM_LINE);
                  } else {
                    smoothedRightRef.current = [];
                    rightHandFilter.reset();
                  }
                }

              // 3. Stream to Backend Recognition Engine
              // Word mode: send pristine landmark coordinates (lossless, 2KB vs 50KB JPEG)
              // Fingerspell mode: send JPEG frame (CNN requires raw image input)
              const currentMode = recognitionModeRef.current;
              const sendInterval = currentMode === 'fingerspell' ? 66 : 33; // 15 FPS fingerspell, 30 FPS word
              if (timestamp - lastWsSendTime >= sendInterval) {
                lastWsSendTime = timestamp;
                if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
                  if (currentMode === 'fingerspell') {
                    // Fingerspell: CNN needs raw image frames
                    if (canvasRef.current && video.videoWidth > 0 && video.videoHeight > 0) {
                      const canvas = canvasRef.current;
                      const ctx = canvas.getContext('2d');
                      if (ctx) {
                        const aspectRatio = video.videoWidth / video.videoHeight;
                        const targetWidth = 640;
                        const targetHeight = Math.round(targetWidth / aspectRatio);
                        
                        if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
                            canvas.width = targetWidth;
                            canvas.height = targetHeight;
                        }
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const base64Frame = canvas.toDataURL('image/jpeg', 0.65);
                        socketRef.current.send(JSON.stringify({
                          type: 'frame',
                          frame: base64Frame,
                          target_lang: selectedLanguageRef.current,
                          mode: currentMode
                        }));
                      }
                    }
                  } else {
                    // Word mode: send raw landmarks directly (no JPEG, no re-detection)
                    socketRef.current.send(JSON.stringify({
                      type: 'landmarks',
                      left_hand: clientLeft.length === 21 ? clientLeft : [],
                      right_hand: clientRight.length === 21 ? clientRight : [],
                      pose: clientPose,
                      target_lang: selectedLanguageRef.current,
                      mode: currentMode
                    }));
                  }
                }
              }
              }
            }

            if (isWebcamActive) {
              animationFrameId = requestAnimationFrame(renderAndSend);
            }
          };

          animationFrameId = requestAnimationFrame(renderAndSend);
        })
        .catch((err) => {
          showToast("Camera access failed. Check site permissions.", "error");
          setIsWebcamActive(false);
          setWebcamStatus('off');
        });
    } else {
      setWebcamStatus('off');
    }

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (socketRef.current) {
        socketRef.current.close();
      }
      // Stop media tracks
      if (webcamElement?.srcObject) {
        const stream = webcamElement.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        webcamElement.srcObject = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isWebcamActive, showToast]);

  // 7. Dynamic Word Learning API call
  const handleLearnWord = async () => {
    const word = newWordToLearn.trim();
    if (!word) return;

    setLearningLoading(true);
    setLearningError('');
    setLearningResult(null);

    try {
      const res = await fetch('http://localhost:8000/api/recognize/learn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word })
      });

      if (res.ok) {
        const data = await res.json() as { data?: LearnedWord };
        setLearningResult(data.data ?? null);
        showToast(`Successfully learned "${word}"!`, "success");
      } else {
        const errData = await res.json() as { detail?: string };
        const errMsg = errData.detail || "Word lookup failed";
        setLearningError(errMsg);
        showToast(errMsg, "error");
      }
    } catch {
      const connErr = "Connection error while learning new word";
      setLearningError(connErr);
      showToast(connErr, "error");
    } finally {
      setLearningLoading(false);
    }
  };

  // TTS active state
  const [isSpeaking, setIsSpeaking] = useState(false);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);

  // Multilingual Text-To-Speech function (default stock voice)
  const handleTextToSpeech = useCallback((textToSpeak: string, langOverride?: string) => {
    if (!textToSpeak || typeof window === 'undefined') return;

    const currentLang = langOverride || selectedLanguage || "en";

    // 1. Immediately cancel and stop any ongoing speech or audio
    try {
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current.src = "";
        currentAudioRef.current = null;
      }
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
    } catch {
      // ignore
    }

    setIsSpeaking(true);

    // 2. Primary Reliable Audio Proxy Stream (default stock voice)
    try {
      const ttsProxyUrl = `http://localhost:8000/api/translate/tts?text=${encodeURIComponent(textToSpeak)}&lang=${currentLang}`;
      const audio = new Audio(ttsProxyUrl);
      currentAudioRef.current = audio;

      audio.onended = () => {
        setIsSpeaking(false);
        currentAudioRef.current = null;
      };

      audio.onerror = (e) => {
        console.warn("Audio proxy playback error:", e);
        setIsSpeaking(false);
      };
      
      audio.play().catch((err) => {
        console.warn("Audio play blocked or interrupted:", err);
        setIsSpeaking(false);
      });
    } catch (err) {
      console.warn("TTS initialization failed:", err);
      setIsSpeaking(false);
    }
  }, [selectedLanguage]);



  // Ensure browser TTS voices are loaded (some browsers load them asynchronously)
  useEffect(() => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return;
    window.speechSynthesis.getVoices();
    const handleVoicesChanged = () => window.speechSynthesis.getVoices();
    window.speechSynthesis.addEventListener('voiceschanged', handleVoicesChanged);
    return () => window.speechSynthesis.removeEventListener('voiceschanged', handleVoicesChanged);
  }, []);

  return (
    <>
      {showIntro && (
        <IntroVideoOverlay
          onComplete={handleDismissIntro}
          isFadingOut={introFadingOut}
        />
      )}
      <div className={`main-dashboard ${showIntro ? 'entering' : 'active'}`} style={{ paddingBottom: '12px', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Offline Reconnection Banner */}
      {!backendOnline && (
        <div className="offline-banner">
          <AlertCircle size={16} />
          <span>Connection to Samvada backend lost. Retrying automatically...</span>
          <button 
            onClick={async () => {
              showToast("Retrying connection manually...", "info");
              try {
                const res = await fetch('http://localhost:8000/');
                if (res.ok) {
                  setBackendOnline(true);
                  showToast("Backend reconnected successfully!", "success");
                  const langRes = await fetch('http://localhost:8000/api/languages');
                  const langData = await langRes.json() as { languages?: SupportedLanguage[] };
                  setLanguages(langData.languages || []);
                  fetchDiagnostics();
                } else {
                  showToast("Backend is still unreachable.", "warning");
                }
              } catch {
                showToast("Backend is offline. Is the start script active?", "error");
              }
            }}
            className="glass-btn"
            style={{ padding: '4px 10px', fontSize: '11px', border: '1px solid rgba(255,255,255,0.15)' }}
          >
            Retry Now
          </button>
        </div>
      )}


      {/* Sidebar */}
      {sidebarOpen && (
        <aside className="sidebar">
            <button className="sidebar-close" onClick={() => setSidebarOpen(false)} title="Close Menu">
              <X size={18} />
            </button>
            
            {/* Sidebar Header: Logo + Brand */}
            <div className="sidebar-header">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/samvada_logo.png" alt="Samvada Logo" className="sidebar-logo" />
              <h1 className="sidebar-brand">SAMVADA</h1>
              <p className="sidebar-tagline">AI Sign Language &amp; Multilingual Translation Engine</p>
            </div>

            {/* Navigation */}
            <nav className="sidebar-nav">
              <button 
                className={`sidebar-nav-item ${activeTab === 'textToSign' ? 'active' : ''}`} 
                onClick={() => { setActiveTab('textToSign'); setSidebarOpen(false); }}
              >
                <User size={18} /> Text / Voice → Sign
              </button>
              <button 
                className={`sidebar-nav-item ${activeTab === 'signToText' ? 'active' : ''}`} 
                onClick={() => { setActiveTab('signToText'); setSidebarOpen(false); }}
              >
                <Video size={18} /> Sign → Text / Translation
              </button>
            </nav>

            {/* About Section */}
            <div className="sidebar-section">
              <h3 className="sidebar-section-title">About Samvada</h3>
              
              <div className="sidebar-about-card">
                <h4><Info size={14} /> What is Samvada?</h4>
                <p>
                  Samvada is an AI-powered real-time sign language translation platform 
                  that bridges the communication gap between deaf and hearing communities. 
                  It converts spoken language to ASL sign videos, and recognises sign 
                  language gestures back into text — across 50+ languages.
                </p>
              </div>

              <div className="sidebar-about-card">
                <h4><Eye size={14} /> Our Vision</h4>
                <p>
                  A world where communication barriers no longer exist. We envision 
                  technology that empowers every individual to express themselves freely, 
                  regardless of hearing ability or spoken language.
                </p>
              </div>

              <div className="sidebar-about-card">
                <h4><Target size={14} /> Our Mission</h4>
                <p>
                  To build the most accurate, fastest, and most accessible AI sign 
                  language translation engine — available to everyone, everywhere. 
                  We leverage cutting-edge LLMs, computer vision, and speech AI to 
                  deliver near-instant multilingual translation.
                </p>
              </div>

              <div className="sidebar-about-card">
                <h4><Heart size={14} /> Our Motive</h4>
                <p>
                  Over 70 million deaf people worldwide rely on sign language as their 
                  primary communication method. Most digital tools ignore them. Samvada 
                  exists to change that — making inclusion not just an ideal, but a 
                  technology-driven reality.
                </p>
              </div>
            </div>

            {/* Footer */}
            <div className="sidebar-footer">
              <p>Built with ❤️ for accessibility</p>
              <p style={{ marginTop: '4px' }}>Samvada © 2026</p>
            </div>
          </aside>
      )}

      {/* Main Content Wrapper (Shifts when sidebar is open) */}
      <div style={{
        marginLeft: sidebarOpen ? '340px' : '0',
        width: sidebarOpen ? 'calc(100% - 340px)' : '100%',
        transition: 'margin-left 0.35s cubic-bezier(0.16, 1, 0.3, 1), width 0.35s cubic-bezier(0.16, 1, 0.3, 1)',
        display: 'flex',
        flexDirection: 'column',
        flex: 1
      }}>
      {/* 1. Header — Hamburger | Logo Center | Controls Right */}
      <header style={{
        margin: '10px 12px 8px',
        padding: '10px 16px',
        display: 'grid',
        gridTemplateColumns: 'auto 1fr auto',
        alignItems: 'center',
        background: 'rgba(255,255,255,0.03)',
        backdropFilter: 'blur(20px)',
        borderRadius: '12px',
        border: '1px solid rgba(255,255,255,0.07)',
      }}>
        {/* Left: Hamburger menu */}
        <button 
          onClick={() => setSidebarOpen(true)} 
          title="Open Menu"
          className="glass-btn"
          style={{ padding: '8px', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)' }}
        >
          <Menu size={18} />
        </button>

        {/* Center: Logo + SAMVADA */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img 
            src="/samvada_logo.png" 
            alt="Samvāda" 
            style={{ height: '36px', width: '36px', objectFit: 'contain', filter: 'drop-shadow(0 2px 8px rgba(212, 168, 67, 0.3))' }} 
          />
          <h1 style={{ fontFamily: 'var(--font-heading)', fontSize: '18px', fontWeight: 700, letterSpacing: '3px', color: '#f8fafc' }}>
            SAMVADA
          </h1>
        </div>

        {/* Right: Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', justifyContent: 'flex-end' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px' }}>
            <span style={{
              width: '7px', height: '7px', borderRadius: '50%',
              background: backendOnline ? '#10b981' : '#ef4444',
              boxShadow: backendOnline ? '0 0 8px #10b981' : '0 0 8px #ef4444'
            }} />
            <span style={{ color: 'var(--color-text-muted)', fontWeight: 500, fontFamily: 'var(--font-inter)' }}>
              {backendOnline ? 'Online' : 'Offline'}
            </span>
          </div>

          <button 
            onClick={() => { setIntroFadingOut(false); setShowIntro(true); }}
            className="glass-btn"
            style={{ padding: '5px 10px', fontSize: '11px', fontFamily: 'var(--font-inter)', borderColor: 'rgba(255,255,255,0.08)' }}
            title="Replay SAMVADA Intro Video"
          >
            ▶ Intro
          </button>

          {backendOnline && (
            <button 
              onClick={() => setShowDiagnostics(!showDiagnostics)}
              className="glass-btn"
              style={{ padding: '5px 10px', fontSize: '11px', fontFamily: 'var(--font-inter)', borderColor: 'rgba(255,255,255,0.08)' }}
            >
              Diagnostics {showDiagnostics ? '▲' : '▼'}
            </button>
          )}

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Globe size={14} color="var(--color-text-muted)" />
            <select 
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="glass-input"
              style={{ padding: '5px 10px', fontSize: '11px', fontFamily: 'var(--font-inter)' }}
            >
              <option value="en">English</option>
              {languages.map((l) => (
                <option key={l.code} value={l.code}>{l.name}</option>
              ))}
            </select>
          </div>
        </div>
      </header>

      {/* System Diagnostics Panel */}
      {showDiagnostics && backendOnline && (
        <div className="glass-panel" style={{
          margin: '0 16px 12px 16px',
          padding: '20px',
          background: 'rgba(10, 8, 24, 0.45)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          transform: 'none'
        }}>
          <h3 style={{ fontSize: '13px', fontWeight: 600, marginBottom: '16px', color: 'var(--color-accent-blue)', letterSpacing: '0.5px' }}>
            SYSTEM DIAGNOSTICS & PIPELINE LATENCY
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
            {/* Operations Latency */}
            {Object.entries(diagnosticsData).map(([opName, metric]) => (
              <div key={opName} className="glass-panel" style={{ padding: '16px', background: 'rgba(0,0,0,0.15)', border: '1px solid rgba(255,255,255,0.03)', transform: 'none' }}>
                <span style={{ fontSize: '10px', color: 'var(--color-text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  {opName.replace(/_/g, ' ')}
                </span>
                <div style={{ fontSize: '20px', fontWeight: 700, margin: '8px 0 4px 0' }}>
                  {metric.average_latency_ms.toFixed(1)} <span style={{ fontSize: '12px', fontWeight: 500, color: 'var(--color-text-muted)' }}>ms</span>
                </div>
                <span style={{ fontSize: '11px', color: metric.success_rate > 0.9 ? '#10b981' : '#ef4444' }}>
                  Success: {(metric.success_rate * 100).toFixed(0)}% ({metric.count} runs)
                </span>
              </div>
            ))}
            {/* Models Status */}
            {Object.entries(modelStates).map(([modelName, state]) => (
              <div key={modelName} className="glass-panel" style={{ padding: '16px', background: 'rgba(0,0,0,0.15)', border: '1px solid rgba(255,255,255,0.03)', transform: 'none' }}>
                <span style={{ fontSize: '10px', color: 'var(--color-text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  {modelName.replace(/_/g, ' ')}
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: '12px 0 0 0' }}>
                  <span style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: state.available ? '#10b981' : '#ef4444',
                    boxShadow: state.available ? '0 0 8px #10b981' : '0 0 8px #ef4444'
                  }} />
                  <span style={{ fontSize: '13px', fontWeight: 500 }}>
                    {state.available ? 'Active & Ready' : 'Unavailable'}
                  </span>
                </div>
                {state.error && (
                  <span style={{ fontSize: '10px', color: 'var(--color-text-muted)', display: 'block', marginTop: '6px', lineHeight: 1.3 }}>
                    {state.error}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Mode navigation */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '10px',
        margin: '6px 0 8px'
      }}>
        <button 
          onClick={() => setActiveTab('textToSign')}
          className={`glass-btn ${activeTab === 'textToSign' ? 'glass-btn-primary' : ''}`}
          style={{ width: '220px', fontFamily: 'var(--font-heading)' }}
        >
          <User size={18} />
          Text / Voice → Sign
          <span className="hotkey-badge">[1]</span>
        </button>
        <button 
          onClick={() => setActiveTab('signToText')}
          className={`glass-btn ${activeTab === 'signToText' ? 'glass-btn-primary' : ''}`}
          style={{ width: '220px', fontFamily: 'var(--font-heading)' }}
        >
          <Video size={18} />
          Sign → Text / Translation
          <span className="hotkey-badge">[2]</span>
        </button>
      </div>

      {/* Main Container Grid */}
      <main className="main-grid">
        
        {/* TAB 1: TEXT -> SIGN */}
        {activeTab === 'textToSign' && (
          <>
            {/* Input card */}
            <section className="glass-panel glass-card-blue" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px', minHeight: '600px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 600, fontFamily: 'var(--font-heading)' }}>Translation Input</h2>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button 
                    id="btn-mic-toggle"
                    onClick={handleMicrophoneToggle}
                    className={`glass-btn ${isRecording ? 'animate-glow' : ''}`}
                    style={{
                      padding: '8px 14px',
                      borderRadius: '20px',
                      background: isRecording ? '#ef4444' : 'rgba(255,255,255,0.05)',
                      border: 'none',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px'
                    }}
                  >
                    {isRecording ? <MicOff size={18} color="#fff" /> : <Mic size={18} color="var(--color-text)" />}
                    <span className="hotkey-badge">[M]</span>
                  </button>
                </div>
              </div>

              {/* Language Selection Row */}
              <div style={{ display: 'flex', gap: '12px', width: '100%' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1 }}>
                  <label style={{ fontSize: '11px', color: 'var(--color-text-muted)', fontWeight: 600 }}>TRANSLATE FROM</label>
                  <select 
                    value={sourceLanguage}
                    onChange={(e) => setSourceLanguage(e.target.value)}
                    className="glass-input"
                    style={{ padding: '8px 12px', fontSize: '12px', width: '100%' }}
                  >
                    <option value="auto">🌍 Auto-detect Language</option>
                    <option value="en">English</option>
                    {languages.map((l) => (
                      <option key={l.code} value={l.code}>{l.name}</option>
                    ))}
                  </select>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1 }}>
                  <label style={{ fontSize: '11px', color: 'var(--color-text-muted)', fontWeight: 600 }}>TRANSLATE TO</label>
                  <select 
                    value={selectedLanguage}
                    onChange={(e) => setSelectedLanguage(e.target.value)}
                    className="glass-input"
                    style={{ padding: '8px 12px', fontSize: '12px', width: '100%' }}
                  >
                    <option value="en">English</option>
                    {languages.map((l) => (
                      <option key={l.code} value={l.code}>{l.name}</option>
                    ))}
                  </select>
                </div>
              </div>

              {isRecording ? (
                <div className="waveform">
                  <div className="wave-bar" />
                  <div className="wave-bar" />
                  <div className="wave-bar" />
                  <div className="wave-bar" />
                  <div className="wave-bar" />
                  <div className="wave-bar" />
                  <div className="wave-bar" />
                  <div className="wave-bar" />
                  <div className="wave-bar" />
                  <div className="wave-bar" />
                </div>
              ) : (
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Type a sentence or long paragraphs here... Or click the mic to speak."
                  className="glass-input"
                  style={{ minHeight: '300px', resize: 'none', fontSize: '15px', lineHeight: 1.6, flex: 1 }}
                />
              )}

              <button 
                onClick={() => translateTextToSign()}
                disabled={translating}
                className="glass-btn glass-btn-primary"
                style={{ width: '100%' }}
              >
                {translating ? 'Translating...' : 'Translate to Sign Language'}
                <Send size={16} />
              </button>

              {/* Gloss results */}
              {glosses.length > 0 && (
                <div style={{ marginTop: '16px' }}>
                  <h3 style={{ fontSize: '13px', color: 'var(--color-text-muted)', marginBottom: '8px' }}>ASL Gloss Sequence:</h3>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {glosses.map((g, idx) => {
                      const activeIndex = currentVideoIndex;
                      let bg = 'rgba(255,255,255,0.06)';
                      let color = 'var(--color-text)';
                      let glow = 'none';
                      if (idx < activeIndex) {
                        color = 'var(--color-text-muted)';
                        bg = 'transparent';
                      } else if (idx === activeIndex) {
                        bg = 'rgba(212, 168, 67, 0.2)';
                        color = '#d4a843';
                        glow = '0 0 10px rgba(212, 168, 67, 0.5)';
                      }
                      return (
                      <span key={idx} style={{
                        background: bg,
                        color: color,
                        padding: '4px 10px',
                        borderRadius: '6px',
                        fontSize: '12px',
                        border: '1px solid rgba(255,255,255,0.08)',
                        boxShadow: glow,
                        fontWeight: 600,
                        transition: 'all 0.3s ease'
                      }}>
                        {g}
                      </span>
                    )})}
                  </div>
                </div>
              )}
            </section>

            {/* Visualizer output card */}
            <section className="glass-panel glass-card-purple" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px', minHeight: '600px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 600, fontFamily: 'var(--font-heading)' }}>Sign Language Display</h2>
                
                {/* Replay */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  {(glosses.length > 0) && (
                    <button
                      onClick={() => {
                        setCurrentAnimationIndex(0);
                        setCurrentVideoIndex(0);
                        if (translatedCaption) handleTextToSpeech(translatedCaption);
                      }}
                      className="glass-btn"
                      style={{
                        display: 'flex', alignItems: 'center', gap: '6px',
                        padding: '6px 12px', fontSize: '11px', borderRadius: '8px',
                        background: 'rgba(212, 168, 67, 0.2)', border: '1px solid rgba(212, 168, 67, 0.4)',
                        color: '#d4a843', fontWeight: 600
                      }}
                    >
                      <RotateCcw size={13} /> Replay
                    </button>
                  )}
                </div>
              </div>

              {/* View Output box */}
              <div style={{
                flex: 1,
                background: 'rgba(0,0,0,0.3)',
                borderRadius: '16px',
                overflow: 'hidden',
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: '1px solid var(--glass-border)'
              }}>
                {videoSequence.length > 0 ? (
                  <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '460px', aspectRatio: '16 / 9', background: '#06070a', borderRadius: '16px', overflow: 'hidden' }}>
                    <video 
                      ref={videoRef}
                      style={{
                        position: 'absolute', top: 0, left: 0,
                        width: '100%', height: '100%', objectFit: 'contain',
                        display: typeof videoSequence[currentVideoIndex] === 'string' ? 'block' : 'none'
                      }}
                      controls={false}
                      autoPlay
                      muted
                      playsInline
                    />

                    {typeof videoSequence[currentVideoIndex] === 'object' && (videoSequence[currentVideoIndex] as {type: string}).type === 'image' && (
                      <div style={{
                        position: 'absolute', top: 0, left: 0,
                        width: '100%', height: '100%',
                        display: 'flex', flexDirection: 'column',
                        alignItems: 'center', justifyContent: 'center',
                        background: '#06070a', borderRadius: '16px', overflow: 'hidden'
                      }}>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img 
                          src={`http://localhost:8000/${(videoSequence[currentVideoIndex] as unknown as {url: string}).url}`}
                          alt={`ASL Letter ${(videoSequence[currentVideoIndex] as unknown as {word: string}).word}`}
                          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                        />
                        <div style={{
                          position: 'absolute', top: '16px', right: '16px',
                          background: 'linear-gradient(135deg, #d4a843, #e8913a)', color: '#fff',
                          padding: '6px 16px', borderRadius: '10px',
                          fontSize: '14px', fontWeight: 700,
                          boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
                          letterSpacing: '0.5px'
                        }}>
                          ASL Alphabet {(videoSequence[currentVideoIndex] as unknown as {word: string}).word}
                        </div>
                      </div>
                    )}

                    <div style={{
                      position: 'absolute',
                      bottom: '24px',
                      left: '0',
                      right: '0',
                      display: 'flex',
                      justifyContent: 'center',
                      pointerEvents: 'none',
                      zIndex: 10
                    }}>
                      <span style={{
                        background: 'rgba(0,0,0,0.85)',
                        color: '#fff',
                        padding: '8px 24px',
                        borderRadius: '12px',
                        fontSize: '22px',
                        fontWeight: 'bold',
                        border: '1px solid rgba(255,255,255,0.2)',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
                      }}>
                        {typeof videoSequence[currentVideoIndex] === 'object' 
                          ? (videoSequence[currentVideoIndex] as {word: string}).word 
                          : glosses[currentVideoIndex]}
                      </span>
                    </div>
                  </div>
                ) : (
                  <div style={{ color: 'var(--color-text-muted)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                    <Video size={36} color="var(--color-text-muted)" />
                    <span style={{ fontSize: '13px' }}>Translate text to view sign language videos</span>
                  </div>
                )}
              </div>

              {/* Playback Controls */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px', padding: '0 8px', marginTop: '4px' }}>
                <span style={{ fontSize: '12px', color: 'var(--color-text-muted)', fontWeight: 600 }}>SPEED: {playbackSpeed.toFixed(2)}x</span>
                <input 
                  type="range" 
                  min="0.5" 
                  max="3.0" 
                  step="0.1" 
                  value={playbackSpeed} 
                  onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                  style={{ flex: 1, accentColor: '#d4a843', height: '4px', borderRadius: '2px', outline: 'none' }}
                />
                <button 
                  onClick={() => {
                    setCurrentVideoIndex(0);
                    setCurrentAnimationIndex(0);
                  }}
                  className="glass-btn"
                  style={{ padding: '6px 12px', fontSize: '11px', display: 'flex', gap: '6px', alignItems: 'center' }}
                >
                  <RotateCcw size={14} /> Replay
                </button>
              </div>

              {/* Caption display box */}
              <div style={{
                background: 'rgba(255,255,255,0.03)',
                padding: '16px',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.05)',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '11px', color: 'var(--color-text-muted)', fontWeight: 600 }}>TRANSLATION / CAPTIONS</span>
                    {isSpeaking && (
                      <span style={{
                        fontSize: '10px',
                        color: '#10b981',
                        background: 'rgba(16, 185, 129, 0.15)',
                        border: '1px solid rgba(16, 185, 129, 0.3)',
                        padding: '2px 8px',
                        borderRadius: '10px',
                        fontWeight: 600,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}>
                        <Volume2 size={12} /> SPEAKING...
                      </span>
                    )}
                  </div>
                  {translatedCaption && !translating && (
                    <button 
                      onClick={() => handleTextToSpeech(translatedCaption, selectedLanguage)}
                      title="Play Voice Audio"
                      style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: '4px' }}
                    >
                      <Volume2 size={18} color={isSpeaking ? "#10b981" : "var(--color-accent-blue)"} />
                    </button>
                  )}
                </div>
                <div style={{ minHeight: '24px', display: 'flex', alignItems: 'center', width: '100%' }}>
                  {translating ? (
                    <div className="shimmer" />
                  ) : translatedCaption ? (
                    <p style={{ fontSize: '15px', fontWeight: 500 }}>
                      <ProgressiveTextReveal text={translatedCaption} speed={40} />
                    </p>
                  ) : (
                    <p style={{ fontSize: '15px', fontWeight: 500, color: 'var(--color-text-muted)' }}>
                      Waiting for translation input...
                    </p>
                  )}
                </div>
              </div>
            </section>
          </>
        )}

        {/* TAB 2: SIGN -> TEXT */}
        {activeTab === 'signToText' && (
          <>
            {/* Input camera */}
            <section className="glass-panel glass-card-blue" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px', minHeight: '600px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 600, fontFamily: 'var(--font-heading)' }}>Webcam Input</h2>
                
                {/* Mode Selector */}
                <div style={{
                  background: 'rgba(0,0,0,0.2)',
                  padding: '4px',
                  borderRadius: '10px',
                  display: 'flex',
                  gap: '4px',
                  border: '1px solid var(--glass-border)'
                }}>
                  <button 
                    onClick={() => setRecognitionMode('word')}
                    className="glass-btn" 
                    style={{
                      padding: '6px 12px',
                      fontSize: '11px',
                      borderRadius: '8px',
                      background: recognitionMode === 'word' ? 'rgba(255,255,255,0.1)' : 'transparent',
                      border: 'none'
                    }}
                  >
                    Words Mode
                  </button>
                  <button 
                    onClick={() => setRecognitionMode('fingerspell')}
                    className="glass-btn"
                    style={{
                      padding: '6px 12px',
                      fontSize: '11px',
                      borderRadius: '8px',
                      background: recognitionMode === 'fingerspell' ? 'rgba(255,255,255,0.1)' : 'transparent',
                      border: 'none'
                    }}
                  >
                    Fingerspell Mode
                  </button>
                  <button 
                    onClick={() => setRecognitionMode('sentence')}
                    className="glass-btn"
                    style={{
                      padding: '6px 12px',
                      fontSize: '11px',
                      borderRadius: '8px',
                      background: recognitionMode === 'sentence' ? 'rgba(255,255,255,0.1)' : 'transparent',
                      border: 'none'
                    }}
                  >
                    Sentence Mode
                  </button>
                </div>
              </div>

              {/* Webcam stream window - Cinema-Grade Expansive Viewport */}
              <div style={{
                background: 'rgba(0,0,0,0.35)',
                borderRadius: '20px',
                minHeight: '580px',
                maxHeight: '720px',
                width: '100%',
                flex: 1,
                aspectRatio: '16 / 9',
                overflow: 'hidden',
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: '1px solid var(--glass-border)',
                boxShadow: '0 20px 50px rgba(0,0,0,0.5)'
              }}>
                {isWebcamActive ? (
                  <>
                    <video 
                      ref={webcamRef}
                      style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }}
                      playsInline
                      muted
                      autoPlay
                    />
                    {/* Hidden canvas for extraction - 640x480 resolution for high accuracy */}
                    <canvas ref={canvasRef} width={640} height={480} style={{ display: 'none' }} />
                    {/* Landmark Reticle Wireframe Overlay */}
                    <canvas ref={overlayCanvasRef} className="overlay-canvas" width={224} height={224} style={{ display: showWireframe ? 'block' : 'none', transform: 'scaleX(-1)' }} />

                    {/* Live Status Overlay */}
                    <div style={{
                      position: 'absolute',
                      top: '16px',
                      left: '16px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      background: 'rgba(10, 8, 24, 0.75)',
                      backdropFilter: 'blur(10px)',
                      padding: '6px 12px',
                      borderRadius: '8px',
                      border: '1px solid rgba(255,255,255,0.08)',
                      fontSize: '11px',
                      fontWeight: 600,
                      pointerEvents: 'none'
                    }}>
                      {webcamStatus === 'connected' ? (
                        <>
                          <span style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            background: '#10b981',
                            boxShadow: '0 0 8px #10b981',
                            animation: 'pulse-glow 1.5s infinite alternate'
                          }} />
                          <span style={{ color: '#10b981', letterSpacing: '0.5px' }}>LIVE SCANNING</span>
                        </>
                      ) : (
                        <>
                          <div className="spinner" style={{ width: '10px', height: '10px', borderWidth: '1.5px' }} />
                          <span style={{ color: 'var(--color-text-muted)', letterSpacing: '0.5px' }}>CONNECTING...</span>
                        </>
                      )}
                    </div>

                    {/* Wireframe Overlay Toggle Control */}
                    <button
                      onClick={() => setShowWireframe(!showWireframe)}
                      style={{
                        position: 'absolute',
                        bottom: '16px',
                        right: '16px',
                        background: 'rgba(10, 8, 24, 0.75)',
                        backdropFilter: 'blur(10px)',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: '8px',
                        padding: '4px 10px',
                        fontSize: '11px',
                        color: showWireframe ? '#d4a843' : 'var(--color-text-muted)',
                        cursor: 'pointer'
                      }}
                    >
                      Wireframe: {showWireframe ? 'ON' : 'OFF'}
                    </button>
                  </>
                ) : (
                  <div style={{ color: 'var(--color-text-muted)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                    <Video size={36} color="var(--color-text-muted)" />
                    <span style={{ fontSize: '13px' }}>Click Start Camera to recognize gestures</span>
                  </div>
                )}
              </div>

              {webcamError && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#ef4444', fontSize: '12px' }}>
                  <AlertCircle size={16} />
                  <span>{webcamError}</span>
                </div>
              )}

              <button 
                onClick={() => setIsWebcamActive(!isWebcamActive)}
                className={`glass-btn ${isWebcamActive ? 'glass-btn-primary' : ''}`}
                style={{ width: '100%' }}
              >
                {isWebcamActive ? 'Stop Webcam' : 'Start Webcam'}
                <span className="hotkey-badge">[Space]</span>
              </button>
            </section>

            {/* Output results */}
            <section className="glass-panel glass-card-purple" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px', minHeight: '600px' }}>
              <h2 style={{ fontSize: '18px', fontWeight: 600, fontFamily: 'var(--font-heading)' }}>Translation Output</h2>
              
              {/* English output */}
              <div style={{
                background: 'rgba(0,0,0,0.2)',
                padding: '20px',
                borderRadius: '16px',
                border: '1px solid var(--glass-border)',
                minHeight: '220px',
                flex: 1
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ fontSize: '11px', color: 'var(--color-text-muted)', fontWeight: 600 }}>ENGLISH GRAMMAR OUTPUT</span>
                  {recognizedText && (
                    <button 
                      onClick={() => handleTextToSpeech(recognizedText)}
                      style={{ background: 'transparent', border: 'none', cursor: 'pointer' }}
                    >
                      <Volume2 size={16} color="var(--color-accent-blue)" />
                    </button>
                  )}
                </div>
                <p style={{ fontSize: '18px', fontWeight: 500, letterSpacing: '0.3px' }}>
                  {recognizedText ? (
                    <ProgressiveTextReveal text={recognizedText} speed={40} />
                  ) : (
                    <span style={{ color: 'var(--color-text-muted)' }}>Waiting for webcam input gestures...</span>
                  )}
                </p>
              </div>

              {/* Multilingual Translation output */}
              {selectedLanguage !== 'en' && (
                <div style={{
                  background: 'rgba(0,0,0,0.2)',
                  padding: '20px',
                  borderRadius: '16px',
                  border: '1px solid var(--glass-border)',
                  minHeight: '220px',
                  flex: 1
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span style={{ fontSize: '11px', color: 'var(--color-text-muted)', fontWeight: 600 }}>
                      TRANSLATION ({languages.find(l => l.code === selectedLanguage)?.name.toUpperCase()})
                    </span>
                    {translatedOutput && (
                      <button 
                        onClick={() => handleTextToSpeech(translatedOutput)}
                        style={{ background: 'transparent', border: 'none', cursor: 'pointer' }}
                      >
                        <Volume2 size={16} color="var(--color-accent-purple)" />
                      </button>
                    )}
                  </div>
                  <p style={{ fontSize: '18px', fontWeight: 500, letterSpacing: '0.3px' }}>
                    {translatedOutput ? (
                      <ProgressiveTextReveal text={translatedOutput} speed={40} />
                    ) : (
                      <span style={{ color: 'var(--color-text-muted)' }}>Waiting for webcam input gestures...</span>
                    )}
                  </p>
                </div>
              )}

              {/* Gloss List */}
              {recognizedGlosses.length > 0 && (
                <div>
                  <h3 style={{ fontSize: '12px', color: 'var(--color-text-muted)', marginBottom: '8px' }}>Raw Gloss List:</h3>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {recognizedGlosses.map((g, idx) => (
                      <span key={idx} style={{
                        background: 'rgba(255,255,255,0.06)',
                        padding: '4px 10px',
                        borderRadius: '6px',
                        fontSize: '12px',
                        border: '1px solid rgba(255,255,255,0.08)',
                        fontWeight: 600
                      }}>
                        {g}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </section>
          </>
        )}

      </main>

      {/* Dynamic Word Learning Drawer Section */}
      <section className="glass-panel" style={{
        margin: '8px 12px',
        padding: '24px 32px',
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.05)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '24px' }}>
          <div style={{ background: 'rgba(245, 158, 11, 0.1)', padding: '10px', borderRadius: '12px' }}>
            <BookOpen size={28} color="var(--color-accent-amber)" />
          </div>
          <div>
            <h2 style={{ fontSize: '20px', fontWeight: 600, fontFamily: 'var(--font-heading)' }}>Dynamic Dictionary</h2>
            <p style={{ fontSize: '13px', color: 'var(--color-text-muted)', marginTop: '4px' }}>Learn unknown words online instantly</p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '16px', maxWidth: '600px', marginBottom: '24px' }}>
          <input
            type="text"
            value={newWordToLearn}
            onChange={(e) => setNewWordToLearn(e.target.value)}
            placeholder="Type any word (e.g. 'audacious')"
            className="glass-input"
            style={{ flex: 1, padding: '12px 16px', fontSize: '15px' }}
          />
          <button 
            onClick={handleLearnWord}
            disabled={learningLoading}
            className="glass-btn"
            style={{ borderColor: 'var(--color-accent-amber)', color: 'var(--color-accent-amber)', padding: '12px 24px', fontSize: '15px' }}
          >
            <Search size={18} />
            {learningLoading ? 'Searching...' : 'Learn Word'}
          </button>
        </div>

        {learningError && (
          <div style={{ color: '#ef4444', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertCircle size={16} />
            <span>{learningError}</span>
          </div>
        )}

        {learningResult && (
          <div style={{
            background: 'rgba(0,0,0,0.15)',
            padding: '24px',
            borderRadius: '16px',
            border: '1px solid rgba(255,255,255,0.05)',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontSize: '20px', fontWeight: 700, color: 'var(--color-accent-amber)' }}>
                {newWordToLearn.toLowerCase()}
              </span>
              {learningResult.phonetic && (
                <span style={{ fontSize: '13px', color: 'var(--color-text-muted)', fontStyle: 'italic' }}>
                  [{learningResult.phonetic}]
                </span>
              )}
            </div>

            {learningResult.meanings.map((meaning, idx) => (
              <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--color-accent-blue)', textTransform: 'uppercase' }}>
                  {meaning.partOfSpeech}
                </span>
                <ul style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {meaning.definitions?.map((def: string, dIdx: number) => (
                    <li key={dIdx} style={{ fontSize: '13px', color: 'var(--color-text)' }}>
                      {def}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
            
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '11px',
              color: '#10b981',
              marginTop: '10px'
            }}>
              <CheckCircle size={14} />
              <span>Word added to dictionary cache successfully. Fingerspelling enabled for mapping.</span>
            </div>
          </div>
        )}
      </section>
      </div>

      {/* 8. Stacking Toast Container */}
      <div className="toast-container">
        {toasts.map(toast => (
          <div key={toast.id} className={`toast toast-${toast.type}`}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: toast.type === 'success' ? '#10b981' : toast.type === 'error' ? '#ef4444' : toast.type === 'warning' ? '#f59e0b' : '#d4a843',
              boxShadow: toast.type === 'success' ? '0 0 8px #10b981' : toast.type === 'error' ? '0 0 8px #ef4444' : toast.type === 'warning' ? '0 0 8px #f59e0b' : '0 0 8px #d4a843'
            }} />
            <span style={{ fontSize: '13px', fontWeight: 500 }}>{toast.message}</span>
          </div>
        ))}
      </div>
      </div>
    </>
  );
}
        
