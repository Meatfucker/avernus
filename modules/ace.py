from modules.acestep.pipeline_ace_step import ACEStepPipeline


async def generate_ace(avernus_pipeline,
                       prompt,
                       lyrics,
                       actual_seeds=None,
                       guidance_scale=None,
                       audio_duration=None,
                       infer_step=None,
                       omega_scale=None):
    kwargs = {}
    kwargs["prompt"] = prompt
    kwargs["lyrics"] = lyrics
    kwargs["guidance_scale"] = guidance_scale if guidance_scale is not None else 15.0
    kwargs["audio_duration"] = audio_duration if audio_duration is not None else 60.0
    kwargs["infer_step"] = infer_step if infer_step is not None else 60
    kwargs["omega_scale"] = omega_scale if omega_scale is not None else 10.0
    kwargs["save_path"] = "./tests/test.wav"
    kwargs["manual_seeds"] = [str(actual_seeds)]

    avernus_pipeline = await load_ace_pipeline(avernus_pipeline)

    try:
        avernus_pipeline.pipeline(**kwargs)
        return "./tests/test.wav"
    except Exception as e:
        print(f"generate_ace ERROR: {e}")
        return None

async def load_ace_pipeline(avernus_pipeline):
    if avernus_pipeline.model_type != "ace":
        print("loading AceStepPipeline")
        await avernus_pipeline.delete_pipeline()
        generator = ACEStepPipeline("models/", dtype="bfloat16")
        await avernus_pipeline.set_pipeline(generator, "ace")
    return avernus_pipeline