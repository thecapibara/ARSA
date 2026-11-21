/**
 * Procedural Web Audio API synthesizer for ARSA tactical SFX.
 * Zero external assets — synthesizes gunshot cracks, physical impacts,
 * and obstacle placement sounds on demand.
 */
export class TacticalAudio {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private compressor: DynamicsCompressorNode | null = null;
  private noiseBuffer: AudioBuffer | null = null;
  private muted = false;

  /**
   * Unlocks and resumes the AudioContext on a user gesture.
   */
  public unlock(): void {
    const ctx = this.ensureContext();
    if (ctx && ctx.state === 'suspended') {
      void ctx.resume();
    }
  }

  public isMuted(): boolean {
    return this.muted;
  }

  public setMuted(muted: boolean): void {
    this.muted = muted;
    if (this.masterGain && this.ctx) {
      this.masterGain.gain.setValueAtTime(
        muted ? 0 : 0.72,
        this.ctx.currentTime
      );
    }
  }

  public toggleMute(): boolean {
    this.setMuted(!this.muted);
    return this.muted;
  }

  /**
   * Crisp high-snap 9mm pistol crack + short room tail.
   */
  public playPistol(): void {
    const ctx = this.activeContext();
    if (!ctx) return;

    const now = ctx.currentTime;

    // Sharp transient snap
    this.triggerNoiseBurst(ctx, now, 0.065, {
      filterType: 'bandpass',
      freqStart: 2400,
      freqEnd: 750,
      q: 0.9,
      gain: 0.85,
    });

    // Punchy mid-low barrel pop
    this.triggerPitchSweep(ctx, now, 0.06, {
      wave: 'triangle',
      freqStart: 310,
      freqEnd: 62,
      gain: 0.55,
    });

    // Short indoor chamber tail
    this.triggerNoiseBurst(ctx, now + 0.012, 0.24, {
      filterType: 'lowpass',
      freqStart: 1200,
      freqEnd: 240,
      q: 0.7,
      gain: 0.28,
    });
  }

  /**
   * Deep punchy 12ga shotgun boom + layered scatter crack.
   */
  public playShotgun(): void {
    const ctx = this.activeContext();
    if (!ctx) return;

    const now = ctx.currentTime;

    // Sub-bass chest boom
    this.triggerPitchSweep(ctx, now, 0.19, {
      wave: 'sine',
      freqStart: 145,
      freqEnd: 34,
      gain: 0.95,
    });

    // Broad primary blast
    this.triggerNoiseBurst(ctx, now, 0.17, {
      filterType: 'bandpass',
      freqStart: 1400,
      freqEnd: 260,
      q: 0.65,
      gain: 0.9,
    });

    // Layered pellet scatter cracks
    const scatterOffsets = [0.014, 0.03, 0.048];
    for (let i = 0; i < scatterOffsets.length; i++) {
      this.triggerNoiseBurst(ctx, now + scatterOffsets[i], 0.055, {
        filterType: 'bandpass',
        freqStart: 2500 - i * 320,
        freqEnd: 850,
        q: 1.4,
        gain: 0.36 - i * 0.07,
      });
    }

    // Long reverberant tail
    this.triggerNoiseBurst(ctx, now + 0.02, 0.45, {
      filterType: 'lowpass',
      freqStart: 950,
      freqEnd: 160,
      q: 0.6,
      gain: 0.42,
    });
  }

  /**
   * Tight 5.56 rifle supersonic crack + metallic bolt transient.
   */
  public playRifle(): void {
    const ctx = this.activeContext();
    if (!ctx) return;

    const now = ctx.currentTime;

    // Supersonic crack
    this.triggerNoiseBurst(ctx, now, 0.055, {
      filterType: 'highpass',
      freqStart: 1900,
      freqEnd: 900,
      q: 0.8,
      gain: 0.88,
    });

    // High-velocity pressure whip
    this.triggerPitchSweep(ctx, now, 0.048, {
      wave: 'sawtooth',
      freqStart: 1100,
      freqEnd: 130,
      gain: 0.42,
    });

    // Tight mid punch
    this.triggerPitchSweep(ctx, now, 0.085, {
      wave: 'triangle',
      freqStart: 210,
      freqEnd: 52,
      gain: 0.5,
    });

    // Metallic bolt cycle transient
    this.triggerTonePing(ctx, now + 0.036, 0.032, 3450, 0.14);
    this.triggerTonePing(ctx, now + 0.056, 0.028, 4800, 0.1);

    // Crisp tail
    this.triggerNoiseBurst(ctx, now + 0.015, 0.28, {
      filterType: 'lowpass',
      freqStart: 1500,
      freqEnd: 260,
      q: 0.7,
      gain: 0.25,
    });
  }

  /**
   * Dull physical thud / punch / shove impact sound.
   */
  public playImpact(heavy: boolean): void {
    const ctx = this.activeContext();
    if (!ctx) return;

    const now = ctx.currentTime;

    this.triggerPitchSweep(ctx, now, heavy ? 0.14 : 0.11, {
      wave: 'sine',
      freqStart: heavy ? 128 : 96,
      freqEnd: heavy ? 36 : 42,
      gain: heavy ? 0.75 : 0.48,
    });

    this.triggerNoiseBurst(ctx, now, heavy ? 0.085 : 0.11, {
      filterType: 'lowpass',
      freqStart: heavy ? 540 : 340,
      freqEnd: 110,
      q: 0.9,
      gain: heavy ? 0.45 : 0.26,
    });
  }

  /**
   * Subtle low-mid wooden/concrete clack when placing an obstacle box.
   */
  public playBoxDrop(): void {
    const ctx = this.activeContext();
    if (!ctx) return;

    const now = ctx.currentTime;

    this.triggerPitchSweep(ctx, now, 0.06, {
      wave: 'triangle',
      freqStart: 220,
      freqEnd: 105,
      gain: 0.34,
    });

    this.triggerNoiseBurst(ctx, now, 0.045, {
      filterType: 'bandpass',
      freqStart: 620,
      freqEnd: 320,
      q: 2.8,
      gain: 0.24,
    });
  }

  /**
   * Deep sci-fi/kinetic bass whoosh + crisp air pressure burst for Force Wave.
   */
  public playForceWave(): void {
    const ctx = this.activeContext();
    if (!ctx) return;

    const now = ctx.currentTime;

    // Deep kinetic sub-bass drop
    this.triggerPitchSweep(ctx, now, 0.34, {
      wave: 'sine',
      freqStart: 135,
      freqEnd: 26,
      gain: 0.95,
    });

    // Sci-fi harmonic resonance sweep
    this.triggerPitchSweep(ctx, now, 0.24, {
      wave: 'triangle',
      freqStart: 320,
      freqEnd: 46,
      gain: 0.42,
    });

    // Crisp compressed air pressure burst
    this.triggerNoiseBurst(ctx, now, 0.22, {
      filterType: 'bandpass',
      freqStart: 3400,
      freqEnd: 380,
      q: 1.05,
      gain: 0.78,
    });

    // Wide kinetic shockwave whoosh tail
    this.triggerNoiseBurst(ctx, now + 0.018, 0.4, {
      filterType: 'lowpass',
      freqStart: 1100,
      freqEnd: 130,
      q: 0.75,
      gain: 0.44,
    });
  }

  private ensureContext(): AudioContext | null {
    if (this.ctx) return this.ctx;
    if (typeof window === 'undefined') return null;

    const AudioCtx =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!AudioCtx) return null;

    const ctx = new AudioCtx();
    const compressor = ctx.createDynamicsCompressor();
    compressor.threshold.value = -10;
    compressor.knee.value = 12;
    compressor.ratio.value = 6;
    compressor.attack.value = 0.002;
    compressor.release.value = 0.12;

    const masterGain = ctx.createGain();
    masterGain.gain.value = this.muted ? 0 : 0.72;

    compressor.connect(masterGain);
    masterGain.connect(ctx.destination);

    // Pre-generate 1.2s white noise buffer for zero-allocation shots
    const length = Math.floor(ctx.sampleRate * 1.2);
    const buffer = ctx.createBuffer(1, length, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < length; i++) {
      data[i] = Math.random() * 2 - 1;
    }

    this.ctx = ctx;
    this.compressor = compressor;
    this.masterGain = masterGain;
    this.noiseBuffer = buffer;
    return ctx;
  }

  private activeContext(): AudioContext | null {
    if (this.muted) return null;
    const ctx = this.ensureContext();
    if (!ctx) return null;
    if (ctx.state === 'suspended') {
      void ctx.resume();
    }
    return ctx;
  }

  private triggerNoiseBurst(
    ctx: AudioContext,
    startTime: number,
    duration: number,
    opts: {
      filterType: BiquadFilterType;
      freqStart: number;
      freqEnd: number;
      q: number;
      gain: number;
    }
  ): void {
    if (!this.noiseBuffer || !this.compressor) return;

    const src = ctx.createBufferSource();
    src.buffer = this.noiseBuffer;

    const filter = ctx.createBiquadFilter();
    filter.type = opts.filterType;
    filter.Q.setValueAtTime(opts.q, startTime);
    filter.frequency.setValueAtTime(opts.freqStart, startTime);
    filter.frequency.exponentialRampToValueAtTime(
      Math.max(30, opts.freqEnd),
      startTime + duration
    );

    const env = ctx.createGain();
    env.gain.setValueAtTime(0.0001, startTime);
    env.gain.linearRampToValueAtTime(opts.gain, startTime + 0.003);
    env.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);

    src.connect(filter);
    filter.connect(env);
    env.connect(this.compressor);

    src.start(startTime);
    src.stop(startTime + duration + 0.02);
  }

  private triggerPitchSweep(
    ctx: AudioContext,
    startTime: number,
    duration: number,
    opts: {
      wave: OscillatorType;
      freqStart: number;
      freqEnd: number;
      gain: number;
    }
  ): void {
    if (!this.compressor) return;

    const osc = ctx.createOscillator();
    osc.type = opts.wave;
    osc.frequency.setValueAtTime(opts.freqStart, startTime);
    osc.frequency.exponentialRampToValueAtTime(
      Math.max(20, opts.freqEnd),
      startTime + duration
    );

    const env = ctx.createGain();
    env.gain.setValueAtTime(0.0001, startTime);
    env.gain.linearRampToValueAtTime(opts.gain, startTime + 0.002);
    env.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);

    osc.connect(env);
    env.connect(this.compressor);

    osc.start(startTime);
    osc.stop(startTime + duration + 0.02);
  }

  private triggerTonePing(
    ctx: AudioContext,
    startTime: number,
    duration: number,
    freq: number,
    gain: number
  ): void {
    if (!this.compressor) return;

    const osc = ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, startTime);

    const env = ctx.createGain();
    env.gain.setValueAtTime(0.0001, startTime);
    env.gain.linearRampToValueAtTime(gain, startTime + 0.0015);
    env.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);

    osc.connect(env);
    env.connect(this.compressor);

    osc.start(startTime);
    osc.stop(startTime + duration + 0.015);
  }
}
