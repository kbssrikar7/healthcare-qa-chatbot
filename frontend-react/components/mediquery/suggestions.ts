export interface SuggestionItem {
  text: string;
  description: string;
  prompt: string;
}

export const UI_SUGGESTIONS: SuggestionItem[] = [
  {
    text: "Symptoms of Type 2 Diabetes",
    description: "Recognize early warning signs",
    prompt: "What are the symptoms of Type 2 Diabetes?",
  },
  {
    text: "Treatment for Hypertension",
    description: "Guidelines & medication options",
    prompt: "What is the first-line treatment for hypertension?",
  },
  {
    text: "Side effects of Dolo 650",
    description: "Paracetamol safety profile",
    prompt: "What are the side effects of Dolo 650?",
  },
  {
    text: "Causes of acute migraine",
    description: "Triggers & pathophysiology",
    prompt: "What causes acute migraine?",
  },
];
