import { test, expect } from "@playwright/test";

test.describe("Chat Page", () => {
  test("should display chat interface", async ({ page }) => {
    await page.goto("/");

    // Check header is visible
    await expect(page.locator("h1")).toContainText("chatbot-local");

    // Check admin link exists
    await expect(page.getByRole("link", { name: "Admin" })).toBeVisible();

    // Check collection selector exists
    await expect(page.locator("select")).toBeVisible();

    // Check input field exists
    await expect(page.getByPlaceholder("Posez votre question...")).toBeVisible();

    // Check send button exists
    await expect(page.getByRole("button", { name: "Envoyer" })).toBeVisible();
  });

  test("should navigate to admin page", async ({ page }) => {
    await page.goto("/");

    await page.getByRole("link", { name: "Admin" }).click();

    await expect(page).toHaveURL("/admin");
    await expect(page.locator("h1")).toContainText("Admin");
  });

  test("should show welcome message when no messages", async ({ page }) => {
    await page.goto("/");

    await expect(page.getByText("Bienvenue!")).toBeVisible();
  });
});

test.describe("Admin Page", () => {
  test("should display admin interface", async ({ page }) => {
    await page.goto("/admin");

    // Check header
    await expect(page.locator("h1")).toContainText("Admin");

    // Check back link
    await expect(page.getByRole("link", { name: /Retour au chat/ })).toBeVisible();

    // Check collections section
    await expect(page.getByText("Collections")).toBeVisible();

    // Check create collection input
    await expect(page.getByPlaceholder("Nouvelle collection...")).toBeVisible();
  });

  test("should navigate back to chat", async ({ page }) => {
    await page.goto("/admin");

    await page.getByRole("link", { name: /Retour au chat/ }).click();

    await expect(page).toHaveURL("/");
  });

  test("should create a new collection", async ({ page }) => {
    await page.goto("/admin");

    const collectionName = `test-${Date.now()}`;

    // Type collection name
    await page.getByPlaceholder("Nouvelle collection...").fill(collectionName);

    // Click create button
    await page.getByRole("button", { name: "Créer" }).click();

    // Wait for success message or collection to appear
    await expect(page.getByText(collectionName)).toBeVisible({ timeout: 10000 });
  });
});
