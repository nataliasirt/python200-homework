# Week 8 Warmup

## Cloud Concepts

### Cloud Concepts Question 1

The core economic model of cloud computing is pay-as-you-go. Instead of buying and maintaining your own servers up front, you rent computing resources when you need them and pay based on usage, which reduces large upfront costs and shifts the work of maintaining hardware to the cloud provider.

### Cloud Concepts Question 2

Vertical scaling means making one machine more powerful by adding resources like CPU, RAM, or a better GPU, while horizontal scaling means adding more machines to share the workload. You might choose vertical scaling for a single model training job that needs a stronger GPU, and horizontal scaling for a web app or batch system where the work can be distributed across multiple servers.

Horizontal scaling applies to the web app after the viral launch because adding more servers helps handle a much larger number of users at the same time.

Vertical scaling applies to the model training job because the goal is to make one machine stronger with more RAM and a faster GPU.

Horizontal scaling applies to the data pipeline because the files can be split across multiple machines and processed in parallel.

### Cloud Concepts Question 3

Gmail is SaaS because you use a finished software product through the browser and do not manage the infrastructure behind it.

Azure Virtual Machines is IaaS because Azure gives you the virtual hardware, but you still manage the operating system and what runs on it.

Azure App Service is PaaS because the platform handles the server infrastructure while you focus on deploying and configuring your app.

AWS S3 is IaaS because it provides raw cloud storage infrastructure that you use and organize, rather than a complete end-user application.

GitHub Codespaces is PaaS because it gives you a managed development environment without requiring you to manage the underlying machines yourself.

Snowflake is SaaS because it is a fully managed data platform that you use as a finished service instead of running the underlying system yourself.

IaaS is cloud infrastructure like virtual machines, storage, and networking that you rent and configure yourself. An example is Azure Virtual Machines, where you are responsible for managing the operating system, installed software, app code, and most of the system configuration.

PaaS is a managed platform where the cloud provider handles more of the infrastructure so you can focus on your application. An example is Azure App Service, where you manage your code, app settings, and deployment, while the provider manages the servers and much of the runtime environment.

SaaS is a complete software product delivered over the internet that you mostly just use. An example is Gmail, where you manage your account, content, and how you use the software, but Google manages the application, servers, and infrastructure.

### Cloud Concepts Question 4

A managed data platform like Databricks or Snowflake is a cloud-based system for storing, querying, and analyzing data without having to build and manage all the underlying infrastructure yourself. Compared with using Azure directly, you gain convenience, built-in tools, and less operational work, but you give up some control, flexibility, and usually pay more for that abstraction.

### Cloud Concepts Question 5

The cloud is probably not the right choice when you need extremely specialized hardware or low-level control that a cloud platform does not expose well. It is also probably not the right choice when your workload is very steady and predictable, so owning your own infrastructure may be cheaper over time.

## Azure Basics

### Azure Basics Question 1

An Azure subscription is the top-level billing and access container for cloud resources, while a resource group is a smaller logical container used to organize related resources inside a subscription. Your personal resource group is yours alone, while the CTD course shares the subscription.

### Azure Basics Question 2

Ephemeral means the Cloud Shell environment itself can be reset, so local session state and files do not automatically last forever between sessions. In this course setup, persistence comes from attaching Cloud Shell to Azure storage so your files are saved there instead of only in the temporary shell environment.

### Azure Basics Question 3

Your SSH private key is the secret key that stays on your machine, while your SSH public key is the shareable key that gets uploaded to remote systems. Uploading the public key is safe because it can be used to verify that you own the matching private key, but it cannot be used to reconstruct the private key itself.

### Azure Basics Question 4

Cloud Shell output from `az account show`:

```json
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "[redacted]",
    "type": "user"
  }
}
```

When you add `--output table`, Azure CLI formats the same information as a cleaner human-readable table instead of raw JSON.
