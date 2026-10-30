#!/bin/bash
# Build OsuHands.app and build/hands_selftest with the Command Line Tools (no Xcode).
#
#   ./build.sh                    ad-hoc signature (TCC grants are tied to this exact binary)
#   SIGN_ID="OsuHands Local" ./build.sh   sign with a keychain identity instead; grants then
#                                 survive rebuilds (designated requirement = identifier + cert)
#
# The bundle is only touched when the compiled binary or Info.plist changed, so a rebuild of
# unchanged sources keeps the same cdhash and the Screen Recording / Accessibility grants.
# Never runs the app. Launch it with: open -g OsuHands.app
set -euo pipefail
cd "$(dirname "$0")"

SDK=$(xcrun --sdk macosx --show-sdk-path)
FLAGS=(-O -swift-version 5 -target arm64-apple-macos26.0 -sdk "$SDK")
APP=OsuHands.app
mkdir -p build "$APP/Contents/MacOS"

swiftc "${FLAGS[@]}" -module-name OsuHands -o build/OsuHands.new \
  Sources/Common/*.swift Sources/App/*.swift \
  -framework AppKit -framework ScreenCaptureKit -framework CoreMedia -framework CoreVideo -framework ApplicationServices
swiftc "${FLAGS[@]}" -module-name HandsSelfTest -o build/hands_selftest \
  Sources/Common/*.swift Sources/SelfTest/*.swift

changed=0
if ! cmp -s build/OsuHands.new build/OsuHands || [[ ! -f "$APP/Contents/MacOS/OsuHands" ]]; then
  mv build/OsuHands.new build/OsuHands
  cp build/OsuHands "$APP/Contents/MacOS/OsuHands"
  changed=1
else
  rm build/OsuHands.new
fi
if ! cmp -s Info.plist "$APP/Contents/Info.plist"; then
  cp Info.plist "$APP/Contents/Info.plist"
  changed=1
fi
# A local self-signed identity (signing/, made once with openssl + security) keeps the designated
# requirement at "identifier + certificate leaf", so TCC grants survive rebuilds.
KC="$PWD/signing/osuhands.keychain-db"
if [[ -z "${SIGN_ID:-}" && -f "$KC" ]]; then SIGN_ID="OsuHands Local Signing"; fi
if [[ $changed == 1 ]] || ! codesign --verify --strict "$APP" 2>/dev/null; then
  if [[ "${SIGN_ID:-}" == "OsuHands Local Signing" ]]; then
    security unlock-keychain -p osuhands "$KC"
    ORIG=$(security list-keychains -d user | tr -d '"' | xargs)
    security list-keychains -d user -s $ORIG "$KC"   # codesign only finds identities on the search list
    trap 'security list-keychains -d user -s $ORIG' EXIT
  fi
  codesign --force --sign "${SIGN_ID:--}" --identifier com.osujev.hands --timestamp=none "$APP"
  echo "signed $APP (${SIGN_ID:-ad-hoc})"
else
  echo "$APP unchanged: same signature, existing permission grants still apply"
fi
codesign --verify --strict --verbose=1 "$APP"
codesign -dv "$APP" 2>&1
codesign -d -r- "$APP" 2>&1 | grep designated
