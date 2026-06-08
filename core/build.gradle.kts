group = "io.tensors4j"
version = project.findProperty("VERSION_NAME") ?: "0.7.0-SNAPSHOT"

plugins {
    id("java")
}

repositories{
    mavenCentral()
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}

tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:6.0.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")

    // ND4J for benchmark comparisons
    implementation("org.nd4j:nd4j-native-platform:1.0.0-M2.1")
}