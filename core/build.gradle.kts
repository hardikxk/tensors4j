group = "io.github.hardikxk"
version = project.findProperty("VERSION_NAME") ?: "0.7.0-SNAPSHOT"

plugins {
    id("java")
    `maven-publish`
    signing
}

repositories{
    mavenCentral()
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<Javadoc>().configureEach {
    val standardOptions = options as StandardJavadocDocletOptions
    standardOptions.addBooleanOption("Xdoclint:none", true)
    standardOptions.addStringOption("-add-modules", "jdk.incubator.vector")
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
    testImplementation(platform("org.junit:junit-bom:6.1.1"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")

    // ND4J for benchmark comparisons
    implementation("org.nd4j:nd4j-native-platform:1.0.0-M2.1")
}

// Maven Central Publishing
java {
    withSourcesJar()
    withJavadocJar()
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])

            pom {
                name.set("tensors4j-core")
                description.set("Core module for the tensors4j library")
                url.set("https://github.com/hardikxk/tensors4j")

                licenses {
                    license {
                        name.set("The Apache License, Version 2.0")
                        url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                    }
                }
                developers {
                    developer {
                        id.set("hardikxk")
                        name.set("Hardik Kumar")
                        email.set("hardikxk@gmail.com")
                    }
                }
                scm {
                    connection.set("scm:git:git://github.com/hardikxk/tensors4j.git")
                    developerConnection.set("scm:git:ssh://github.com/hardikxk/tensors4j.git")
                    url.set("https://github.com/hardikxk/tensors4j")
                }
            }
        }
    }

    repositories {
        maven {
            name = "Staging"
            url = uri(layout.buildDirectory.dir("staging-repository"))
        }
    }
}

signing {
    val signingKey = System.getenv("GPG_PRIVATE_KEY")
    val signingPassphrase = System.getenv("GPG_PASSPHRASE")

    if (!signingKey.isNullOrEmpty()) {
        useInMemoryPgpKeys(signingKey, signingPassphrase)
        sign(publishing.publications["mavenJava"])
    }
}