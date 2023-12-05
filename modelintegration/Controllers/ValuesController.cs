using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace modelintegration.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ValuesController : ControllerBase
    {
        private InferenceSession _session;
        public ValuesController(InferenceSession session)
        {
            _session = session;
        }
        // GET api/values
        [HttpGet]
        public ActionResult<IEnumerable<string>> Get()
        {
            return new string[] { "value1", "value2" };
        }

        // GET api/values/5
        [HttpGet("{id}")]
        public ActionResult<string> Get(int id)
        {
            return "value";
        }

        // POST api/values
        [HttpPost]
        public void Post([FromBody] string value)
        {
        }

        // PUT api/values/5
        [HttpPut("{id}")]
        public void Put(int id, [FromBody] string value)
        {
        }

        // DELETE api/values/5
        [HttpDelete("{id}")]
        public void Delete(int id)
        {
        }

        [HttpPost]
        [Route("Score")]
        public ActionResult Score([FromBody]HousingData data)
        {
             var inputs1 = new List<NamedOnnxValue>()
              {
                  NamedOnnxValue.CreateFromTensor<float>("float_input", data.AsTensor())
              };
            var result = _session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("float_input", data.AsTensor())


            });
            //var score = result.First().Value.AsEnumerable<float>().ToArray(); 

            //var prediction = new Prediction { PredictedValue = score.First() * 100000 };
           // result.Dispose();
            //foreach(var r in result)
            //{

            //}
            //Prediction pred = new Prediction();
            //string output = string.Empty;
            //Object obj = new Object();
            // using (var outputs1 = _session.Run(inputs1))
            //{
            //    // get intermediate value
            //    var input2 = outputs1.First();
            //    var score = outputs1.First().AsTensor<int>().ToDenseTensor().Buffer.ToArray();
            //};
                //obj = input2;
                // pred.PredictedValue = score.First();
                //output = input2.Value;

             
            //    Tensor<float> score = result.First().AsTensor<float>();
            //Prediction pred = new Prediction();
            //pred.PredictedValue = score.First();
            //result.Dispose();
            
            return Ok(result);
        }
    }
}

public class HousingData
{
    public int Age { get; set; }
    public float Income { get; set; }
    public int Family { get; set; }
    public float CCAvg { get; set; }
    public float Education { get; set; }
    
    public float SecurityAccount { get; set; }
    public float CDAccount { get; set; }
    public float online { get; set; }

    public int CreditCard { get; set; }
    public Tensor<float> AsTensor()
    {
        float[] data = new float[]
        {
            Age, Income, Family, CCAvg,
            Education, SecurityAccount, CDAccount,online , CreditCard
        };
        int[] dimensions = new int[] { 1, 9 };
        return new DenseTensor<float>(data, dimensions);
    }
}
public class Prediction
{
    public float PredictedValue { get; set; }
}
