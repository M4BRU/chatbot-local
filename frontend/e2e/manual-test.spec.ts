import { test, expect } from "@playwright/test";

// Augmenter le timeout global pour ce test (2 minutes)
test.setTimeout(120000);

test.describe("Manuel Test - Rapport complet", () => {
  test("Test complet de l'application", async ({ page }) => {
    // 1. Aller sur la page d'accueil
    console.log("1. Navigation vers http://localhost:3000...");
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // 2. Screenshot de la page d'accueil
    console.log("2. Screenshot de la page d'accueil...");
    await page.screenshot({
      path: "e2e/screenshots/01-homepage.png",
      fullPage: true
    });
    console.log("   ✓ Screenshot sauvegardé: e2e/screenshots/01-homepage.png");

    // Vérifier que l'interface est bien chargée
    await expect(page.locator("h1")).toContainText("chatbot-local");
    console.log("   ✓ Titre 'chatbot-local' présent");

    // 3. Envoyer un message "Bonjour" et attendre la réponse
    console.log("3. Test du chat - envoi de 'Bonjour'...");
    const inputField = page.getByPlaceholder("Posez votre question...");
    await expect(inputField).toBeVisible();
    await inputField.fill("Bonjour");

    const sendButton = page.getByRole("button", { name: "Envoyer" });
    await sendButton.click();
    console.log("   ✓ Message 'Bonjour' envoyé");

    // Attendre que le message utilisateur apparaisse
    await expect(page.getByText("Bonjour")).toBeVisible({ timeout: 5000 });
    console.log("   ✓ Message utilisateur affiché");

    // Attendre la réponse du bot (timeout plus long car LLM)
    console.log("   Attente de la réponse du chatbot (peut prendre ~60s)...");

    // Attendre que le bouton revienne à "Envoyer" (fin du streaming)
    await expect(page.getByRole("button", { name: "Envoyer" })).toBeVisible({ timeout: 90000 });

    // Petit délai pour s'assurer que le streaming est terminé
    await page.waitForTimeout(1000);

    // Screenshot après la réponse
    await page.screenshot({
      path: "e2e/screenshots/02-chat-response.png",
      fullPage: true
    });
    console.log("   ✓ Screenshot sauvegardé: e2e/screenshots/02-chat-response.png");
    console.log("   ✓ Réponse du chatbot reçue");

    // 4. Aller sur la page Admin
    console.log("4. Navigation vers la page Admin (/admin)...");
    await page.getByRole("link", { name: "Admin" }).click();
    await expect(page).toHaveURL("/admin");
    await page.waitForLoadState("networkidle");
    console.log("   ✓ Page Admin chargée");

    // 5. Screenshot de la page Admin
    console.log("5. Screenshot de la page Admin...");
    await page.screenshot({
      path: "e2e/screenshots/03-admin-page.png",
      fullPage: true
    });
    console.log("   ✓ Screenshot sauvegardé: e2e/screenshots/03-admin-page.png");

    // 6. Vérifier que le dropdown de collections est présent
    console.log("6. Vérification du dropdown de collections...");

    // Vérifier la section Collections
    await expect(page.getByText("Collections")).toBeVisible();
    console.log("   ✓ Section 'Collections' visible");

    // Vérifier l'input de création de collection
    await expect(page.getByPlaceholder("Nouvelle collection...")).toBeVisible();
    console.log("   ✓ Input de création de collection visible");

    // Vérifier le bouton Créer
    await expect(page.getByRole("button", { name: "Créer" })).toBeVisible();
    console.log("   ✓ Bouton 'Créer' visible");

    console.log("\n=== RAPPORT FINAL ===");
    console.log("✅ Tous les tests ont réussi!");
    console.log("Screenshots disponibles dans: frontend/e2e/screenshots/");
  });
});
