import { useState, useEffect, useRef } from 'react';
import api from '@/api';
import logo from '@/assets/images/alkhidmat.png';
import { auth } from '../firebase';
import { RecaptchaVerifier, signInWithPhoneNumber } from "firebase/auth";

function UserLogin({ onLoginSuccess, onBack }) {
  const [phoneNumber, setPhoneNumber] = useState('');
  const [otp, setOtp] = useState('');
  const [step, setStep] = useState('phone'); // 'phone' or 'otp'
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isLegacyAuth, setIsLegacyAuth] = useState(false);
  const confirmationResultRef = useRef(null);
  const recaptchaVerifierRef = useRef(null);

  // useEffect(() => {
  //   // Initialize reCAPTCHA verifier only if auth is available
  //   if (auth && !recaptchaVerifierRef.current) {
  //     try {
  //       recaptchaVerifierRef.current = new RecaptchaVerifier(auth, 'recaptcha-container', {
  //         'size': 'invisible',
  //         'callback': (response) => {
  //           console.log('[FIREBASE] reCAPTCHA solved');
  //         }
  //       });
  //     } catch (err) {
  //       console.error('[FIREBASE] Recaptcha initialization failed:', err);
  //     }
  //   }

  //   return () => {
  //     if (recaptchaVerifierRef.current) {
  //       recaptchaVerifierRef.current.clear();
  //       recaptchaVerifierRef.current = null;
  //     }
  //   };
  // }, []);
  useEffect(() => {
    if (!auth || recaptchaVerifierRef.current) return;

    const setupRecaptcha = async () => {
      try {
        console.log("[FIREBASE] Initializing invisible reCAPTCHA...");
        
        // Cleanup old instance if any
        if (window.recaptchaVerifier) {
          try { window.recaptchaVerifier.clear(); } catch (e) {}
        }

        window.recaptchaVerifier = new RecaptchaVerifier(auth, 'recaptcha-container', {
          'size': 'invisible',
          'callback': (response) => {
            console.log('[FIREBASE] reCAPTCHA solved');
          },
          'expired-callback': () => {
            console.warn('[FIREBASE] reCAPTCHA expired, resetting...');
          }
        });

        await window.recaptchaVerifier.render();
        recaptchaVerifierRef.current = window.recaptchaVerifier;
        console.log("[FIREBASE] reCAPTCHA ready");
      } catch (err) {
        console.error("[FIREBASE] reCAPTCHA init error:", err);
      }
    };

    setupRecaptcha();

    return () => {
      if (recaptchaVerifierRef.current) {
        try { recaptchaVerifierRef.current.clear(); } catch (e) {}
        recaptchaVerifierRef.current = null;
        window.recaptchaVerifier = null;
      }
    };
  }, [auth]);


  // async function handleSendOTP(e) {
  //   e.preventDefault();
  //   if (!phoneNumber) return;
    
  //   setError('');
  //   setLoading(true);

  //   try {
  //     const appVerifier = recaptchaVerifierRef.current;
  //     const result = await signInWithPhoneNumber(auth, phoneNumber, appVerifier);
  //     confirmationResultRef.current = result;
  //     setStep('otp');
  //     console.log('[FIREBASE] OTP Sent successfully');
  //   } catch (err) {
  //     console.error('[FIREBASE] Error sending OTP:', err);
  //     setError(err.message || 'Failed to send OTP. Please check your phone number.');
  //   } finally {
  //     setLoading(false);
  //   }
  // }
async function handleSendOTP(e) {
  e.preventDefault();

  if (!phoneNumber) return;

  setError("");
  setLoading(true);

  try {
    // Format phone number
    let formattedPhone = phoneNumber.replace(/\D/g, "");

    if (formattedPhone.startsWith("0")) {
      formattedPhone = "92" + formattedPhone.slice(1);
    }

    if (!formattedPhone.startsWith("92")) {
      formattedPhone = "92" + formattedPhone;
    }

    formattedPhone = "+" + formattedPhone;

    console.log("[FIREBASE] Sending OTP to:", formattedPhone);

    const appVerifier = recaptchaVerifierRef.current;

    if (!appVerifier) {
      throw new Error("reCAPTCHA not initialized");
    }

    console.log("[DEBUG-AUTH] Calling signInWithPhoneNumber for:", formattedPhone);
    const result = await signInWithPhoneNumber(
      auth,
      formattedPhone,
      appVerifier
    );
    console.log("[DEBUG-AUTH] signInWithPhoneNumber success");

    confirmationResultRef.current = result;

    setPhoneNumber(formattedPhone);

    setStep("otp");

    console.log("[FIREBASE] OTP Sent successfully");
    } catch (err) {
      console.error("[FIREBASE] Error sending OTP, falling back to terminal:", err);
      
      try {
        // Fallback to legacy terminal OTP
        await api.sendOTP(phoneNumber);
        setIsLegacyAuth(true);
        setStep('otp');
        setError("⚠️ Firebase SMS failed. Falling back to Terminal OTP (Developer Mode). Check server logs!");
      } catch (fallbackErr) {
        console.error("[FALLBACK] Terminal OTP failed:", fallbackErr);
        setError(err.message || "Failed to send OTP. Please check your connection.");
      }
    } finally {
      setLoading(false);
    }
  }
  async function handleVerifyOTP(e) {
    e.preventDefault();
    if (!otp) return;
    
    setError('');
    setLoading(true);

    try {
      let response;
      
      if (isLegacyAuth) {
        // ── Legacy Terminal Verification ──
        console.log('[AUTH] Using legacy terminal verification...');
        response = await api.verifyOTP(phoneNumber, otp);
      } else {
        // ── Firebase Verification ──
        if (!confirmationResultRef.current) throw new Error("No verification session found");
        const result = await confirmationResultRef.current.confirm(otp);
        const user = result.user;
        const idToken = await user.getIdToken();
        console.log('[FIREBASE] Verified successfully, syncing with backend...');
        response = await api.firebaseLogin(idToken);
      }
      
      // Store session
      localStorage.setItem('user_session_id', response.session_id);
      localStorage.setItem('user_id', response.user.id);
      if (response.user.name) {
        localStorage.setItem('user_name', response.user.name);
      } else {
        localStorage.removeItem('user_name'); // Clear old names for new users
      }
      
      onLoginSuccess(response);
    } catch (err) {
      console.error('[AUTH] Login failed:', err);
      setError(err.data?.detail || err.message || 'Invalid OTP or session expired');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 px-4">
      <div className="max-w-md w-full bg-white rounded-2xl shadow-xl p-8">
        {onBack && (
          <button
            onClick={onBack}
            className="mb-4 text-sm text-gray-600 hover:text-gray-800 flex items-center gap-1"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to Welcome
          </button>
        )}
        <div className="text-center mb-8">
          <img src={logo} alt="Alkhidmat" className="w-32 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-gray-800">User Login</h1>
          <p className="text-gray-600 mt-2">Sign in with your phone number</p>
        </div>

        {/* reCAPTCHA container required for Firebase Phone Auth */}
        <div
          id="recaptcha-container"
          className={step === 'phone' ? 'mb-4' : 'hidden'}
          style={{ minHeight: "1px" }}
        ></div>

        {!auth && (
          <div className="mb-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg text-yellow-800 text-sm">
            <strong>⚠️ Firebase not configured</strong>
            <p className="mt-1">Please add your Firebase credentials to the <code>.env</code> file to enable phone login.</p>
          </div>
        )}

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
            {error}
          </div>
        )}

        {step === 'phone' ? (
          <form onSubmit={handleSendOTP} className="space-y-4">
            <div>
              <label htmlFor="phone" className="block text-sm font-medium text-gray-700 mb-2">
                Phone Number
              </label>
              <input
                id="phone"
                type="tel"
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
                placeholder="+92 300 1234567"
                required
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <button
              type="submit"
              disabled={loading || !phoneNumber}
              className="w-full py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Sending...' : 'Send OTP'}
            </button>
          </form>
        ) : (
          <form onSubmit={handleVerifyOTP} className="space-y-4">
            <div>
              <label htmlFor="otp" className="block text-sm font-medium text-gray-700 mb-2">
                Enter OTP
              </label>
              <input
                id="otp"
                type="text"
                value={otp}
                onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                placeholder="123456"
                maxLength={6}
                required
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-center text-2xl tracking-widest"
              />
              <p className="text-sm text-gray-500 mt-2">
                OTP sent to {phoneNumber}
              </p>
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => {
                  setStep('phone');
                  setOtp('');
                  setError('');
                }}
                className="flex-1 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
              >
                Change Number
              </button>
              <button
                type="submit"
                disabled={loading || otp.length !== 6}
                className="flex-1 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? 'Verifying...' : 'Verify'}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}

export default UserLogin;

