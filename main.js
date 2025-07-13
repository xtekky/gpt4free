import Client from '@gpt4free/g4f.dev';

const client = new Client();
const result = await client.chat.completions.create({
    model: 'gpt-4.1',  // Or "gpt-4o", "deepseek-v3"
    messages: [{ role: 'user', content: 'Explain quantum computing' }]
});
process.stdout.write(result.choices[0].message.content);