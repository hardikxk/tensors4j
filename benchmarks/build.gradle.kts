group = "io.tensors4j"
version = "0.7.0-SNAPSHOT"

plugins {
    id("java")
    id("me.champeau.jmh") version "0.7.3"
}

repositories{
    mavenCentral()
}

jmh {
    jvmArgs.addAll(listOf("--enable-preview", "--add-modules", "jdk.incubator.vector"))
    warmupIterations = 2
    iterations = 2
    fork = 2
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

    // core import
    implementation(project(":core"))
    // ND4J for benchmark comparisons
    implementation("org.nd4j:nd4j-native-platform:1.0.0-M2.1")
}
