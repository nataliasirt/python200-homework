# Week 8 Project

## Video Link

https://1drv.ms/v/c/54ffc85db90d581c/IQALdYAJW4nJQIillHRjNW9YAbWQc2Z5qnn5gblzznpKzsE?e=mLnvai

## Portal Walkthrough Notes

In the video, I showed the Azure portal under the Code the Dream tenant, opened my personal resource group, and pointed out the storage account inside it. I also opened Cloud Shell, showed that files in `~/clouddrive` persist when storage is mounted, showed my SSH key files in `~/.ssh`, and ran `az group list --output table` to show the resource groups available in the subscription.

## Cost Analysis Summary

I built two cost scenarios in the Azure Pricing Calculator for East US on Linux. Scenario A was a lightweight setup using a `Standard_B1s` VM for about 160 hours per month. Scenario B was a much heavier analytics setup using a `Standard_NC6s_v3` GPU VM for 730 hours per month, plus an Azure SQL Database in the General Purpose tier with 4 vCores, and Azure Blob Storage with 1 TB of data.

Using current Linux VM rates, the lightweight `Standard_B1s` compute came out to about `$1.66` per month (`$0.0104/hour x 160 hours`). The GPU VM portion of Scenario B came out to about `$2233.80` per month (`$3.06/hour x 730 hours`), which shows how dramatically more expensive GPU compute is than a small basic VM. My full Pricing Calculator totals were:

- Scenario A total: `$[fill in your calculator total]`
- Scenario B total: `$[fill in your calculator total]`

The most surprising part was how quickly the GPU workload cost increased compared with the lightweight VM. Even before adding the SQL Database and Blob Storage charges, the GPU VM alone was already much more expensive than Scenario A. While exploring the calculator, I also noticed how many separate cost levers Azure exposes, including compute size, storage tier, redundancy, and how long a resource runs.

## Script Output

After updating and running `project_08.py` in Cloud Shell, the script printed the monthly VM-only estimates for the two scenarios:

```text
=== Monthly Cost Estimates ===
Scenario A (lightweight):       $1.66
Scenario B (GPU VM only):       $2233.80
Scenario B VM costs 1342.4x more than Scenario A
```

The script output should match the VM-only portion of the Pricing Calculator because it uses the same hourly rates multiplied by the required number of hours. If my calculator totals are different overall, that is expected because Scenario B also includes SQL Database and Blob Storage charges that are not part of the Python script.
