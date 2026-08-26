import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Loaded from signing.properties (gitignored — never commit keystore
// passwords). See project_paperwork/scratch/mobile_port_notes.md
// ("Play Store Readiness") for why the keystore currently referenced here is
// a local placeholder, not the real production signing key.
val signingProps = Properties().apply {
    val f = rootProject.file("signing.properties")
    if (f.exists()) load(f.inputStream())
}

android {
    namespace = "com.mediquery.mobile"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.mediquery.mobile"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"

        // arm64-v8a only: a 1B-param on-device LLM with a GPU delegate isn't
        // realistically usable on 32-bit ARM or x86 hardware, and bundling
        // the ~530MB model makes shipping all 4 ABIs' worth of native libs
        // wasteful. Covers the vast majority of real Android phones from the
        // last ~8 years; excludes emulators (x86/x86_64) and legacy 32-bit
        // devices.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    // Model/KB assets are already-compressed/high-entropy binary formats —
    // compressing them again in the APK wastes build time for no size win,
    // and (more importantly) an asset stored compressed can't be opened via
    // AssetManager.openFd(), which AssetInstaller needs for the upfront
    // total-size progress reporting during the first-run copy.
    androidResources {
        noCompress += listOf("task", "jsonl")
    }

    signingConfigs {
        if (signingProps.containsKey("storeFile")) {
            create("release") {
                storeFile = rootProject.file(signingProps.getProperty("storeFile"))
                storePassword = signingProps.getProperty("storePassword")
                keyAlias = signingProps.getProperty("keyAlias")
                keyPassword = signingProps.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            if (signingProps.containsKey("storeFile")) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")

    val composeBom = platform("androidx.compose:compose-bom:2026.02.00")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.compose.material:material-icons-extended")

    // On-device LLM inference — same library Google AI Edge Gallery uses.
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.11.0")

    // Markdown rendering for chat answers — same library Gallery uses.
    implementation("com.halilibo.compose-richtext:richtext-commonmark:1.0.0-alpha02")
    implementation("com.halilibo.compose-richtext:richtext-ui-material3:1.0.0-alpha02")

    // On-device query embedding (all-MiniLM-L6-v2 ONNX export) for retrieval.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.29.0")
}
