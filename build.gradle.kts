plugins {
    id("java")
    kotlin("jvm")
}

group = "io.tensors4j"
version = project.findProperty("VERSION_NAME") ?: "0.1.0-SNAPSHOT"

tasks.test {
    useJUnitPlatform()
}
dependencies {
    implementation(kotlin("stdlib-jdk8"))
}
repositories {
    mavenCentral()
}
kotlin {
    jvmToolchain(25)
}