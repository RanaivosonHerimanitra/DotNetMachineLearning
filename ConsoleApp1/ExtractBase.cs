using Microsoft.ML;
using Microsoft.ML.Data;
using Samples.Dynamic;
using System.Data.SqlClient;

namespace DotNetMachineLearning
{
    public abstract class ExtractBase : Contexte
    {
        protected virtual object GetDataFromFile<T>(string[] path, char separatorChar, bool hasHeader = true)
        {
            if (String.IsNullOrEmpty(separatorChar.ToString())) { throw new ArgumentNullException(Constantes.DelimiteurDoitExister); };
            if (path == null) { throw new ArgumentNullException(Constantes.FichierDoitExister); };
            if (path.Length == 0) { throw new ArgumentNullException(Constantes.FichierDoitExister); };
            if (path.Length == 1) return MLContext.Data.LoadFromTextFile<T>(path[0], separatorChar, hasHeader);
            var textLoader = MLContext.Data.CreateTextLoader<T>(separatorChar, hasHeader);
            return textLoader.Load(path);
        }

        protected virtual object GetDataFromDataBase<T>(string connectionString, string sqlCommand)
        {
            var dbSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, sqlCommand);
            var loader = MLContext.Data.CreateDatabaseLoader<T>();
            return loader.Load(dbSource);
        }
    }
}
