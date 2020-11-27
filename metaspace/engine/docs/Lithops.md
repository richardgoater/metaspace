# Environment setup

## IBM Cloud

Note that all resources should be created in the same region for best performance, 
but not all regions support all necessary services (COS, Functions, Gen2 VPC). 
METASPACE team members can use `eu-de`/Frankfurt for all services. 

1. Get someone to invite you to the organization, set up your IBMid & login
2. Ensure the organization's account is selected in the menu bar (the drop-down next to "Manage").
3. [Create an IAM API Key](https://cloud.ibm.com/iam/apikeys)
    * **Name:** `dev-<your initials>`
    * Copy this key into `lithops.ibm.iam_api_key` in the sm-engine `config.json`
4. [Create a resource group](https://cloud.ibm.com/account/resource-groups)
    * **Name:** `dev-<your initials>`
5. [Create an Object Storage Service](https://cloud.ibm.com/objectstorage/create)
    * **Plan:** Standard
    * **Resource group:** `dev-<your initials>`
6. Create buckets in the service you just created. Use "Customize your bucket", not one of the predefined buckets:
    * A persistent storage bucket for data like imzml files and the centroids cache
        * **Name:** `metaspace-dev-<your initials>-data`:
        * This will be used for normal persistent data such as imzml files, moldb files and the centroids cache
        * **Resiliency:** Regional
        * **Storage class:** Smart Tier
    * A temp bucket for Lithops and other temp data  
        * It's easy to accidentally generate huge amounts of data with Lithops, so this includes a rule to automatically delete objects after 1 day:
        * **Name:** `metaspace-dev-<your initials>-temp`:
        * **Resiliency:** Regional
        * **Storage class:** Standard
        * Create an **Expiration** rule:
            * **Rule name:** Cleanup
            * **Prefix filter:** (leave empty)
            * **Expiration days:** 1
            * Click **Save** above the rule
    * Copy the above bucket names into your sm-engine config.json
        * Set `lithops.lithops.storage_bucket` to the **temp** bucket name
        * In `lithops.sm_storage`, each entry has the pattern `"data type": ["bucket", "prefix"]`. 
            * Set `"pipeline_cache"`'s bucket to the **temp** bucket name
            * Set all other data types' buckets to the **data** bucket name
7. Create Cloud Object Storage service credentials
    * **Role:** Writer
    * **Advanced Options -> Include HMAC Credential:** On
    * (HMAC credentials are slightly faster & more reliable with Lithops as they don't have a token that needs to be 
    continually refreshed)
    * Open the credential details after they're created (click the arrow to the left of the row)
    * Copy the HMAC access key and secret key into the `lithops.ibm_cos` section of your sm-engine `config.json`  
8. Create a [Cloud Functions namespace](https://cloud.ibm.com/functions/) (from the "Current Namespace" menu at the top)
    * **Name:** `dev-<your initials>`
    * **Resource group:** `dev-<your initials>`
    * Once created, open "Namespace settings" and copy the Name and GUID to `lithops.ibm_cf` in the sm-engine `config.json`
9. (Optional) [Set up a VPC](https://cloud.ibm.com/vpc-ext/provision/vs)
    * VPC instances aren't used much in development, so it may be best to just skip this step and share the existing instance. 
    * **Operating system:** Ubuntu
    * **Profile:** Balanced 32GB RAM
    * **SSH Key:** Upload your own SSH public key
    * After creation, go into the instance details and add a Floating IP address
10. Configure your connection to the VPC
    * (If step 8 was skipped) Ask someone to upload your SSH key
    * Start the instance manually through the web UI
    * SSH into the instance to confirm it's set up correctly: `ssh ubuntu@<public IP address>`
    * Stop the instance through the web UI
    * Open the instance details and copy the ID and Floating IP into `lithops.ibm_vpc` in the sm-engine `config.json`
     

## Setting up IBM Cloud CLI

1. Download it from https://cloud.ibm.com/docs/cli
2. Install plugins:
    ```
    ibm_cloud plugin install cloud-functions
    ibm_cloud plugin install cloud-object-storage
    ibm_cloud plugin install vpc-infrastructure
    ```
3. Sign in with `ibmcloud login` and follow the prompts.
4. Use `ibmcloud target --cf` to select the organization / namespace that your functions are in.
5. Use `ibmcloud target -g dev-<your initials>` to select your resource group.
6. If you have already run some invocations, you can use `ibmcloud fn activation list` to list them to confirm that everything is configured correctly

## Debugging strategies

### Viewing Cloud Function logs via CLI

Note: Function logs are only retained by IBM for 14 days, and logs are only made available once the Function has finished running.

Activation IDs can be found in the Lithops debug logs. 

* `ibmcloud fn activation list` lists recent Function calls
* `ibmcloud fn activation list <namespace>` lists recent Function calls in the specified namespace
* `ibmcloud fn activation logs <activation ID>` views the stdout logs of a specific function call
* `ibmcloud fn activation poll` monitors running jobs and prints each job's output when the job finishes

### Running pipeline stages locally

Adding `debug_run_locally=True` to an `executor.map`/`executor.call` will cause it to run the functions in the
calling process. This allows you to easily attach a debugger to the Python process.

Alternatively, the entire pipeline can be run locally by changing the `lithops.lithops.mode` and 
`lithops.lithops.storage_backend` to `localhost`, however this causes Lithops to use `multiprocessing` to run tasks
in parallel in separate processes, which is less useful for debugging.

### Viewing Object Storage data locally

For temporary CloudObjects, the hard part is usually finding where the object is stored - you may have to log its 
location and manually retrieve it from the logs. It may also be necessary to disable `data_cleaner` in the 
Lithops config, so that temporary CloudObjects aren't deleted when the FunctionExecutor is GC'd.

Once you have the bucket & key for the object:

```python
from lithops.storage import Storage
from sm.engine.annotation_lithops.io import deserialize
from sm.engine.util import SMConfig

storage = Storage(lithops_config=SMConfig.get_conf()['lithops'])
obj = deserialize(storage.get_object('bucket', 'key'))
```   

Alternatively, you can use the IBM Cloud CLI to download objects to the local filesystem:

```
ibmcloud cos objects --bucket BUCKET
ibmcloud cos object-get --bucket BUCKET --key KEY DEST_FILENAME
```