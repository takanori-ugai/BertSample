import io.gitlab.arturbosch.detekt.Detekt
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jlleitschuh.gradle.ktlint.reporter.ReporterType

plugins {
    kotlin("jvm") version "2.3.0"
    java
    id("org.jetbrains.dokka") version "2.1.0"
    id("io.gitlab.arturbosch.detekt") version "1.23.8"
    id("com.gradleup.shadow") version "9.3.1"
    id("com.github.jk1.dependency-license-report") version "3.0.1"
    id("org.jlleitschuh.gradle.ktlint") version "14.0.1"
    jacoco
    id("com.vanniktech.maven.publish") version "0.35.0"
    id("com.diffplug.spotless") version "8.2.0"
}

group = "io.github.ugaikit"
version = "0.1"

repositories {
    mavenCentral()
}

dependencies {
    implementation("ai.djl:api:0.36.0")
    implementation("ai.djl.pytorch:pytorch-engine:0.36.0")
    // Select a CUDA build that matches your installed driver/toolkit.
//    implementation("ai.djl.pytorch:pytorch-native-cu124:2.7.1")
    implementation("ai.djl.pytorch:pytorch-jni:2.5.1-0.36.0")
    implementation("ai.djl.pytorch:pytorch-native-cu124:2.5.1")
    implementation("ai.djl.pytorch:pytorch-native-cpu:2.5.1")
    implementation("ai.djl.huggingface:tokenizers:0.36.0")
    implementation("org.slf4j:slf4j-simple:2.0.17")
    implementation("io.github.oshai:kotlin-logging-jvm:7.0.14")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
    testImplementation(kotlin("test"))
}

kotlin {
    jvmToolchain(17)
}

tasks {
    "wrapper"(Wrapper::class) {
        distributionType = Wrapper.DistributionType.ALL
    }

    compileKotlin {
        compilerOptions.jvmTarget.set(JvmTarget.JVM_17)
        doLast { println("Finished compiling Kotlin source code") }
    }

    compileTestKotlin {
        compilerOptions.jvmTarget.set(JvmTarget.JVM_17)
        doLast { println("Finished compiling Kotlin Test source code") }
    }

    compileJava {
        options.encoding = "UTF-8"
        options.compilerArgs.addAll(listOf("-Xlint:deprecation"))
        sourceCompatibility = "17"
        targetCompatibility = "17"
    }

    compileTestJava {
        options.encoding = "UTF-8"
        options.compilerArgs.addAll(listOf("-Xlint:deprecation"))
        sourceCompatibility = "17"
    }

    jacocoTestReport {
        reports {
            xml.required.set(true)
            csv.required.set(true)
            html.outputLocation.set(layout.buildDirectory.dir("jacocoHtml"))
        }
    }

    withType<JacocoReport> {
        dependsOn("test")
        executionData(withType<Test>())
        classDirectories.setFrom(files(listOf("build/classes/kotlin/main")))
        //  sourceDirectories = files(listOf("src/main/java", "src/main/kotlin"))
        sourceDirectories.setFrom(files(listOf("src/main/java", "src/main/kotlin")))
    }

    test {
        useJUnitPlatform()
        testLogging {
//            exceptionFormat = TestExceptionFormat.FULL
            showStandardStreams = true
        }
        // Force CPU execution in tests to avoid CUDA-only native library loading.
//        environment("CUDA_VISIBLE_DEVICES", "-1")
    }

    withType<Detekt>().configureEach {
        // Target version of the generated JVM bytecode. It is used for type resolution.
        jvmTarget = "17"
        reports {
            // observe findings in your browser with structure and code snippets
            html.required.set(true)
            // checkstyle like format mainly for integrations like Jenkins
            xml.required.set(true)
            // similar to the console output, contains issue signature to manually edit baseline files
            txt.required.set(true)
            // standardized SARIF format (https://sarifweb.azurewebsites.net/) to support integrations
            // with Github Code Scanning
            config.setFrom("config/detekt.yml")
            sarif.required.set(true)
        }
    }
}

ktlint {
    version.set("1.8.0")
    verbose.set(true)
    outputToConsole.set(true)
    coloredOutput.set(true)
    reporters {
        reporter(ReporterType.CHECKSTYLE)
        reporter(ReporterType.JSON)
        reporter(ReporterType.HTML)
    }
    filter {
        exclude("**/style-violations.kt")
        exclude("**/ResourceData.kt")
    }
}

spotless {
    java {
        target("src/*/java/**/*.java")
        importOrder()
        removeUnusedImports()

        // Choose one of these formatters.
        googleJavaFormat("1.28.0") // has its own section below
        formatAnnotations() // fixes formatting of type annotations, see below
    }
}

mavenPublishing {
    // Maven Central に公開する場合の設定
    publishToMavenCentral()

    signAllPublications()

    // ライブラリの座標設定
    coordinates("io.github.ugaikit", "bertscore", "0.1.0")

    // POM情報（Maven Centralには必須）
    pom {
        name = "BERTScore4kt"
        description =
            "A Kotlin library for calculating BERTScore using Deep Java Library (DJL) and HuggingFace tokenizers. "
        url = "https://github.com/takanori-ugai/BertScore"
        inceptionYear.set("2026")
        licenses {
            license {
                name = "The Apache License, Version 2.0"
                url = "http://www.apache.org/licenses/LICENSE-2.0.txt"
            }
        }
        developers {
            developer {
                id = "takanori-ugai"
                name = "Takanori Ugai"
                email = "ugai.takanori@gmail.com"
            }
        }
        scm {
            connection = "scm:https://github.com/takanori-ugai/BertScore.git"
            developerConnection = "scm:https://github.com/takanori-ugai/BertScore.git"
            url = "https://github.com/takanori-ugai/BertScore"
        }
    }
}
